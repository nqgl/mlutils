from nqgl.mlutils.components.cache import Cache


import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from nqgl.mlutils.components.linear_reset.actfreq_linear import ActsCache


class CacheLayer(torch.nn.Module):
    def __init__(
        self,
        W: Float[Tensor, "*inst d_in d_out"],
        b_out: Float[Tensor, "*#inst d_out"],
        nonlinearity=torch.nn.ReLU(),
    ):
        super().__init__()
        self.W = W if isinstance(W, nn.Parameter) else nn.Parameter(W)
        # self.b_pre =   b_in if isinstance(b_in, nn.Parameter) else nn.Parameter(b_in)
        self.b = b_out if isinstance(b_out, nn.Parameter) else nn.Parameter(b_out)
        self.nonlinearity = nonlinearity
        self.W = nn.Parameter(W)

    def forward(self, x, cache: Cache):
        cache.x = x
        cache.pre_acts = (pre_acts := (x + self.b_pre) @ self.W + self.b_post)
        cache.acts = (acts := self.nonlinearity(pre_acts))
        return acts


class CacheProcLayer(torch.nn.Module):
    def __init__(self, cachelayer: CacheLayer):
        self.cachelayer = cachelayer
        self.train_cache_template = ActsCache()
        self.eval_cache_template = Cache()
        self.train_process_after_call: set = set()
        self.eval_process_after_call: set = set()

    def forward(self, *x, cache: Cache = None):

        cache = self.prepare_cache(cache)
        acts = self.cachelayer(*x, cache=cache)
        self._update(cache)
        return acts

    def _update(self, cache: Cache):
        if self.training:
            for fn in self.train_process_after_call:
                fn(cache)
        else:
            for fn in self.eval_process_after_call:
                fn(cache)

    def prepare_cache(self, cache: Cache = None):
        if cache is None:
            return self.generate_default_cache()
        return self.register_to_external_cache(cache)

    def generate_default_cache(self):
        if self.training:
            return self.train_cache_template.clone()
        else:
            return self.eval_cache_template.clone()

    def register_to_external_cache(self, cache: Cache):
        cache += self.generate_default_cache()
        return cache
