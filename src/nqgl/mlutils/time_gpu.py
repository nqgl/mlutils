import torch
import time
import inspect

PRINT_DEFAULT = True


class TimedFunc:
    def __init__(self, func, name=None, cpu_time=False, print=PRINT_DEFAULT):
        self.func = func
        self.t = None
        if name is not None:
            self.name = name
        else:
            self.name = (
                func.__name__ if hasattr(func, "__name__") else func.__class__.__name__
            )
        self.times = []
        self.print_on_call = print
        self.__name__ = f"TimedFunc<{self.name}>"
        self.cpu_time = cpu_time

    def __call__(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.time()
        start.record()
        ret = self.func(*args, **kwargs)
        end.record()
        t1 = time.time()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        if self.name is not None and self.print_on_call:
            if self.cpu_time:
                print(self.name, ": ", f"\tGPU={t} \tCPU={t1 - t0}")
            else:
                print(self.name, ": ", t)
        self.times.append(t)
        self.t = t
        return ret

    def __repr__(self):
        return self.__name__


def timedfunc_wrapper(**kwargs):
    return lambda f: TimedFunc(f, **kwargs)


def profilefunc_wrapper(**kwargs):
    return lambda f: ProfileFunc(f, **kwargs)


def time_methods(cls=None, **kwargs):
    if cls is None:
        return lambda c: time_methods(c, **kwargs)
    for name, func in inspect.getmembers(cls, callable):
        if name.startswith("__") and not inspect.isfunction(func):
            continue
        print(name, func)
        setattr(cls, name, TimedFunc(func, name=f"{cls.__name__}::{name}", **kwargs))
    return cls


class ProfileFunc:
    def __init__(
        self,
        func,
        name=None,
        prof_once=True,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ):
        self.func = func
        self.t = None
        self.prof_once = prof_once
        self.times_called = 0
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        if name is not None:
            self.name = name
        else:
            self.name = (
                func.__name__ if hasattr(func, "__name__") else func.__class__.__name__
            )
        self.times = []
        self.print_on_call = print
        self.__name__ = f"ProfiledFunc<{self.name}>"

    def __call__(self, *args, **kwargs):

        if not (self.prof_once and self.times_called > 0):
            try:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.CPU,
                    ],
                    profile_memory=self.profile_memory,
                    record_shapes=self.record_shapes,
                    with_stack=self.with_stack,
                ) as prof:
                    ret = self.func(*args, **kwargs)
            except Exception as e:
                print(f"Exception in {self.name}: {e}")
                prof.export_chrome_trace(f"{self.name}_trace{self.times_called}.json")
                print("saved trace")
                raise e
            prof.export_chrome_trace(f"{self.name}_trace{self.times_called}.json")
            print("saved trace")
            print(
                prof.key_averages(group_by_stack_n=5).table(
                    sort_by="self_cpu_time_total", row_limit=5
                )
            )
            print(
                prof.key_averages(group_by_stack_n=5).table(
                    sort_by="cuda_memory_usage", row_limit=20
                )
            )

        else:
            ret = self.func(*args, **kwargs)
        self.times_called += 1
        return ret

    def test(self, x):
        pass
