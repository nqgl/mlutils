import torch


class TimedFunc(object):
    def __init__(self, func, name=None, print_on_call=False):
        self.func = func
        self.t = None
        if name is not None: 
            self.name = name 
        else:
            self.name = (
                func.__name__ 
                if hasattr(func, "__name__") 
                else func.__class__.__name__
            )
        self.times = []
        self.print_on_call = print_on_call


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
