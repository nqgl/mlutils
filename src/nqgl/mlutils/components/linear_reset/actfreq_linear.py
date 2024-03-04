import torch
from jaxtyping import Float, jaxtyped, Bool
from torch import Tensor
from nqgl.mlutils.components.cache import Cache
from nqgl.mlutils.components.config import WandbDynamicConfig
from nqgl.mlutils.components.cache_layer import CacheLayer, CacheProcLayer
from typing import Any, Union, Optional
from dataclasses import dataclass, field


class ActsCache(Cache):
    acts: Float[Tensor, "batch *inst d_out"] = ...

def raise_nie(fn, ctx):
    def raiser(*args, **kwargs):
        raise NotImplementedError(f"{fn} not implemented in {ctx.__class__.__name__}")
    return raiser


def biastype(size): 
    return Union[Float[Tensor, f"{size}"], Float[Tensor, f"*inst {size}"]]
# removed b_in:Float[Tensor, "*#inst d_in"]
# not useful to have, doesn't need resetting thus not in domain of this module
# make this agnostic to multidimensional or single-dimensional case?
class GenericFreqMaybe:
    def reset_freqs(self, mask=None, initial_activation=0, initial_count=0):
        if mask is None:
            assert isinstance(self.count, int)
            self.active = torch.zeros_like(self.active) + initial_activation
            self.count = initial_count
        else:
            self.active[mask] = initial_activation
            self.count[mask] = initial_count

    def get_dead_neurons(self, count_min, threshold):
        ...


class CountingFreqActMixin(CacheProcLayer):
    def __init__(self):
        self._activations = 0
        self._count = 0
        self.train_cache_template.acts = ...
        # self.train_cache_template.register_write_callback("acts", self._update_count_freqs)
        self.train_process_after_call.add(self._update_count_freqs)


    def steps_counter(self, cache:ActsCache):
        return cache.acts.shape[0]

    def _get_count_freq(self):
        return self._activations / (self._count + 1e-9)

    def _update_count_freqs(self, cache):
        if cache.has.acts:    
            self._activations += self.activated_counter(cache)
            self._count += self.steps_counter(cache)

    def activated_counter(self, cache:ActsCache):
        return cache.acts.count_nonzero(dim=0)

    @property
    def freqs(self):
        return self._get_count_freq()

class EMAFreqMixin:
    ...

@dataclass
class ResettingConfig(WandbDynamicConfig):
    dead_threshold: float = 3e-5




class ResamplingLayer(torch.nn.Module):
    def __init__(self, cfg:ResettingConfig, freq_layer, downstream_weights):
        self.freq_layer = freq_layer
        self.downstream_weights = downstream_weights
        self.T = 0

    def re_init_neurons
    -> rename "resample"

    def resampling_check

    def reset_neurons

    def reset_activation_frequencies
        -> reset_freqs
    
    def get_activation_frequencies(self):
        -> freqs property





class QueuedResettingLinear:
    re_init_queued : callable
    ...
    @torch.no_grad()
    def re_init_neurons(self, x_diff):
        ...


class GhostGradResettingLinear:
    ...

class 




