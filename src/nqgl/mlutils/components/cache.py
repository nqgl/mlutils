from typing import Any
from dataclasses import dataclass, field, Field
from jaxtyping import Float, jaxtyped
from torch import Tensor

# from beartype import beartype as typechecker
from typeguard import typechecked as typechecker

property


# @dataclass


class CacheHas:
    def __init__(self, cache):
        self.cache: Cache = cache

    def __getattribute__(self, __name: str) -> Any:
        if __name == "cache":
            return super().__getattribute__("cache")
        if __name.startswith("_"):
            return super().__getattribute__(__name)
        return self.cache._has(__name)


def cancopy(v):
    return hasattr(v, "copy")


def listdictcopy(dl):
    return {k: (v.copy() if cancopy(v) else v) for k, v in dl.items()}


def listdictadd(da, db, unique=True):
    do = listdictcopy(da)
    for k, v in db.items():
        if k in do:
            summed = do[k] + v
            do[k] = list(dict.fromkeys(summed)) if unique else summed
        else:
            do[k] = v.copy() if cancopy(v) else v
    return do


"""
A different way this could work, is like:
cache.watch.attributenametowatch = True/False
yeah probably I will rewrite this to use Dict[str, Bool]


also possibly todo if this seems useful:
cache[i] -> cache (
    subcache of first cache, default to watch same attributes 
    (make as cache.clone() probably)
)

Cache c:
ci:Cache = c[i]
ci._parent == c
ci._prev = c[i - 1] if i > 0 else None
"""


class Cache:
    """
    Fields are write-once.
    Default value to denote a watched attribute (__NULL_ATTR) is ellipsis
    """

    __RESERVED_NAMES = ["has"]
    _NULL_ATTR = ...
    _unwatched_writes = ...
    _ignored_names = ...
    _write_callbacks = ...
    _lazy_read_funcs = ...
    has: CacheHas

    def __init__(self, callbacks=None, parent=None, subcache_index=None):
        self._NULL_ATTR: Any = ...
        self._unwatched_writes: set = set()
        self._ignored_names: set = set()
        self._write_callbacks: dict = callbacks or {}
        self._lazy_read_funcs: dict = {}
        self._subcaches: dict = {}
        self._parent: Cache = parent
        self._subcache_index = subcache_index
        super().__setattr__("has", CacheHas(self))

    def _watch(self, __name: str = None):
        if __name is None:
            return self._watching(__name)
        if __name in self._lazy_read_funcs:
            raise AttributeError(f"Attribute {__name} is lazy-rendered")
        self.__setattr__(__name, ...)

    def _watching(self, __name: str):
        raise NotImplementedError("Not yet implemented")
        return __name in self.__dict__ and not self._ignored(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.__RESERVED_NAMES:
            raise AttributeError(f"Cannot set reserved attribute {__name}")

        if __name.startswith("_"):
            return super().__setattr__(__name, __value)

        if __value == self._NULL_ATTR:
            if hasattr(self, __name) and not getattr(self, __name) == self._NULL_ATTR:
                raise AttributeError(
                    f"Cache error: Tried to watch attribute {__name}, but {__name} already set to {getattr(self, __name)}"
                )
            else:
                return super().__setattr__(__name, __value)
        self._write(__name, __value)

    def _has(self, __name: str):
        return (
            hasattr(self, __name)
            and super().__getattribute__(__name) != self._NULL_ATTR
        ) or __name in self._lazy_read_funcs

    def _ignored(self, __name: str):
        return __name in self._ignored_names or not hasattr(self, __name)

    def _write(self, __name: str, __value: Any):
        if self._ignored(__name):
            if __name in self._unwatched_writes:
                raise AttributeError(
                    f"Cache overwrite error on unwatched attribute: Unwatched attribute {__name} already written"
                )
            self._unwatched_writes.add(__name)
        elif getattr(self, __name) != self._NULL_ATTR:
            raise AttributeError(
                f"Cache overwrite error: Watched attribute {__name} already set to {getattr(self, __name)}"
            )
        if hasattr(self, __name):
            super().__setattr__(__name, __value)
        if __name in self._write_callbacks:
            for hook in self._write_callbacks[__name]:
                hook(self, __value)

    def _getfields(self):
        values = {}
        watching = set()
        names = {
            name
            for name in self.__dict__
            if not (name.startswith("_") or name in self.__RESERVED_NAMES)
        } - set(self.__class__.__dict__.keys())
        for name in names:
            if self._has(name):
                values[name] = getattr(self, name)
            watching.add(name)
        return watching, values

    def __iadd__(self, other: "Cache"):
        if not isinstance(other, Cache):
            raise TypeError(
                f"Cannot add {other.__class__} to Cache. Must be Cache or subclass"
            )
        o_watching, o_values = other._getfields()
        for watch in o_watching:
            self.__setattr__(watch, ...)
        for name, value in o_values.items():
            self.__setattr__(name, value)
        self._write_callbacks = listdictadd(
            self._write_callbacks, other._write_callbacks
        )
        self._unwatched_writes = self._unwatched_writes.union(other._unwatched_writes)
        self._ignored_names = self._ignored_names.union(other._ignored_names)
        self._lazy_read_funcs = listdictadd(
            self._lazy_read_funcs, other._lazy_read_funcs
        )
        if not other._parent is None:
            raise NotImplementedError("cache copy recieving _parent not yet supported")
        if not other._subcaches == {}:
            raise NotImplementedError(
                "cache copy recieving _subcaches not yet supported"
            )
        assert self._NULL_ATTR == other._NULL_ATTR
        return self

    def __getitem__(self, i):
        if i in self._subcaches:
            return self._subcaches[i]
        else:
            subcache = self.clone()
            subcache._parent = self
            self._subcaches[i] = subcache
            return subcache

    def register_write_callback(self, __name: str, hook, ignore=False):
        if __name.startswith("_"):
            raise AttributeError("Cannot set hook on private attribute")
        if __name in self._write_callbacks:
            self._write_callbacks[__name].append(hook)
        else:
            self._write_callbacks[__name] = [hook]
        if ignore:
            self.add_cache_ignore(__name)

    def add_cache_ignore(self, __name: str):
        if __name.startswith("_"):
            raise AttributeError("Cannot ignore private attribute")
        self._ignored_names.add(__name)

    def clone(self):
        clone = self.__class__()
        clone += self
        return clone

    @property
    def _prev_cache(self):
        if self._parent is None:
            raise AttributeError("No parent cache")
        index = self._subcache_index - 1
        if index not in self._parent._subcaches:
            raise AttributeError("No previous cache")
        return self._parent[index]


def main():
    class TC(Cache):
        tf = ...

    c = Cache()
    tc = TC()
    tc2 = TC()
    import torch

    tc.tf = 3
    tc2.tf = 5
    # TC.tf = 3
    print(tc.tf)
    print(tc2.tf)
    tc.tf = 4
    print(tc.tf)

    # type_example = Float[Tensor, "b c"]
    # # t = type_example(torch.rand(3, 4))
    # t(torch.rand(3, 4), torch.rand(3, 4))
    # t(torch.rand(3, 4), torch.rand(3, 5))

    # t(torch.rand(3, 4), torch.rand(2, 5))

    # print(type_example)


if __name__ == "__main__":
    main()
