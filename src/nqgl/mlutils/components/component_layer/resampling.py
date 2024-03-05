from nqgl.mlutils.components.config import WandbDynamicConfig
from nqgl.mlutils.components.cache import Cache

from nqgl.mlutils.components.component_layer import LayerComponent, ComponentLayer
import torch.nn as nn
from dataclasses import dataclass
import torch
from typing import Optional
from abc import abstractmethod


@dataclass
class ResamplingConfig(WandbDynamicConfig):
    dead_threshold: float = 3e-6
    min_viable_count: int = 10_000
    reset_to_freq: float = 0.001
    reset_to_count: int = 10_000
    check_frequency: int = 100


class ResamplingCache(Cache):
    saved_x: torch.Tensor
    resample: bool
    saved_x_pred: torch.Tensor


class ResamplingMethod(LayerComponent):
    _default_component_name = "resampling_method"
    def __init__(self):
        self.layer: ComponentLayer = None

    def _register_parent_layer(self, layer: ComponentLayer):
        self.layer = layer

    def _update_from_cache(self, cache:Cache, **kwargs):
        if cache.has.resample and cache.resample:
            self.resample_from_cache(cache)

    @abstractmethod
    def resample_from_cache(self, cache: ResamplingCache): ...

class ResamplingComponent(LayerComponent):
    _default_component_name = "resampler"
    train_cache_watch = []
    eval_cache_watch = []

    def __init__(
        self,
        cfg: ResamplingConfig,
        downstream_weights: Optional[nn.Parameter] = None,
    ):
        self.cfg = cfg
        self.downstream_weights = downstream_weights
        self.T = 0
        self.layer: ComponentLayer = None

    def _register_parent_layer(self, layer: ComponentLayer):
        self.layer = layer
        self.layer.train_cache_template.register_write_callback(
            "x", self._resample_hook_x
        )

    def _resample_train_hook_x(self, cache: Cache, x):
        self._step()
        if self.is_resample_step():
            cache.saved_x = ...
            cache.saved_x = x
            cache.resample = ...
            cache.resample = True

    # in the future, can denote this with something like:
    # @ComponentLayer.hook.cache.train.x
    # -> set a class field of any LayerComponent that holds hooks to register
    # and then ComponentLayer is responsible for registering/calling them appropriately
    @torch.no_grad()
    def _resample_train_hook_x_pred(self, cache: ResamplingCache, x_pred):
        if cache.has.resample and cache.resample:
            assert self.is_resample_step()
            cache.saved_x_pred = x_pred

    @abstractmethod
    def _update_from_cache(self, cache: Cache, **kwargs):
        pass

    def is_resample_step(self):
        return self.T % self.cfg.check_frequency == 0

    def _step(self):
        self.T += 1

    def get_dead_neurons(self): ...

    # def re_init_neurons
    # -> rename "resample"

    # def resampling_check

    # def reset_neurons

    # def reset_activation_frequencies
    #     -> reset_freqs

    # def get_activation_frequencies(self):
    #     -> freqs property


class QueuedResettingComponent(ResamplingComponent):
    re_init_queued: callable
    ...

    # was re_init_neurons
    @torch.no_grad()
    def resample_from_x_diff(self, x_diff): ...

    def resample

    # def resample_from_cache(self, cache):
    # self.resample_from_x_diff(cache.x - cache.x_pred)

    @torch.no_grad()
    def proc_x_pred(self, x_pred):
        if T % self.cfg.check_frequency == 0:
            self.re_init_queued(x_pred)


class GhostGradResettingLinear: ...


# class
