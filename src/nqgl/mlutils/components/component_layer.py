from abc import ABC, abstractmethod
from nqgl.mlutils.components.cache import Cache
from nqgl.mlutils.components.cache_layer import CacheLayer, CacheProcLayer
from typing import Dict, Any, List, Union
import torch.nn as nn


class LayerComponent(ABC):
    train_cache_watch: List[str] = []
    eval_cache_watch: List[str] = []
    _default_component_name: str = ...

    @abstractmethod
    def _update_from_cache(self, cache):
        raise NotImplementedError

    @classmethod
    def bind_nonlayer_args(cls, **kwargs):
        return lambda layer: cls(layer=layer, **kwargs)


def islambda(f):
    return callable(f) and getattr(f, "__name__", "") == "<lambda>"


class ComponentLayer(CacheProcLayer):
    def __init__(
        self,
        cfg,
        cachelayer: CacheLayer,
        components: List["LayerComponent"] = [],
        names: Dict["LayerComponent", str] = {},
    ):
        super().__init__(cachelayer)
        self.cfg = cfg
        components = [c(self) if islambda(c) else c for c in components]
        self.components = components
        self.module_components = nn.ModuleList(
            [c for c in components if isinstance(c, nn.Module)]
        )
        self._init_update_watched(components)
        self._init_update_attrs_from_components(components, names)

    def _init_update_watched(self, components):
        for c in components:
            if c.train_cache_watch:
                for name in c.train_cache_watch:
                    self.train_cache_template._watch(name)
            if c.eval_cache_watch:
                for name in c.eval_cache_watch:
                    self.eval_cache_template._watch(name)

    def _init_update_attrs_from_components(self, components, names):
        for c in components:
            if c in names:
                name = names[c]
            else:
                name = c._default_component_name
            if name is None:
                continue
            if hasattr(self, name):
                raise ValueError(f"Component name {name} already exists")
            setattr(self, name, c)

    def _update(self, cache, **kwargs):
        super()._update(cache, **kwargs)
        for c in self.components:
            print(c)
            c._update_from_cache(cache=cache, **kwargs)
