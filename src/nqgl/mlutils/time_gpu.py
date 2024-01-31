import torch


class TimedFunc(object):
    def __init__(self, func, name=None, print=False):
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

    def __call__(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        ret = self.func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        if self.name is not None and self.print_on_call:
            print(self.name, ": ", t)
        self.times.append(t)
        self.t = t
        return ret


def timedfunc_wrapper(**kwargs):
    return lambda f: TimedFunc(f, **kwargs)


class ProfileFunc:
    def __init__(self, func, name, prof_once=True):
        self.func = func
        self.t = None
        self.prof_once = prof_once
        self.times_called = 0
        if name is not None:
            self.name = name
        else:
            self.name = (
                func.__name__ if hasattr(func, "__name__") else func.__class__.__name__
            )
        self.times = []
        self.print_on_call = print
        self.__name__ = f"TimedFunc<{self.name}>"

    def __call__(self, *args, **kwargs):

        if not (self.prof_once and self.times_called > 0):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as prof:
                ret = self.func(*args, **kwargs)
            prof.export_chrome_trace(f"{self.name}_trace{self.times_called}.json")

        else:
            ret = self.func(*args, **kwargs)
        self.times_called += 1
        return ret
