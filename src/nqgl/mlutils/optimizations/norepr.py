class MinRepr():
    def __init__(self, wrapped):
        object.__setattr__(self, "_______wrapped", wrapped)

    def __repr__(self):
        return (
            object.__getattribute__(self, "_______wrapped").__name__ 
            if hasattr(object.__getattribute__(self, "_______wrapped"), "__name__") 
            else object.__getattribute__(self, "_______wrapped").__class__.__name__
        )

    def __getattr__(self, name):
        if name not in ["__repr__"]:
            return getattr(object.__getattribute__(self, "_______wrapped"), name)
        else:
            return object.__getattr__(self, name)

    def __call__(self, *args, **kwargs):
        return object.__getattribute__(self, "_______wrapped")(*args, **kwargs)
    

from functools import partial

def fastpartial(*args, **kwargs): 
    return MinRepr(partial(*args, **kwargs))
