from nqgl.mlutils.components.cache_layer import ActsCache, CacheProcLayer, CacheLayer
from nqgl.mlutils.components.component_layer import (
    LayerComponent,
    ComponentLayer,
)

from nqgl.mlutils.components.config import WandbDynamicConfig
import torch
from abc import ABC, abstractmethod
from typing import List, Type, TypeVar
from dataclasses import dataclass


class FreqTracker(LayerComponent):
    _default_component_name = "activations"

    @abstractmethod
    def __init__(self, parent: CacheProcLayer = None): ...

    @abstractmethod
    def reset_freqs(self, mask=None, initial_activation=0, initial_count=0): ...

    @abstractmethod
    def get_dead_neurons(self, count_min, threshold): ...

    @property
    @abstractmethod
    def freqs(self): ...


@dataclass
class CountingFreqTrackerConfig(WandbDynamicConfig):
    reset_to_freq: float = 0
    reset_to_count: int = 0


class CountingFreqTracker(FreqTracker):
    train_cache_watch = ["acts"]
    eval_cache_watch = []

    def __init__(self, cfg=CountingFreqTrackerConfig()):
        self.num_activations = 0
        self._count = 0
        # self._parent = None
        # self.parent: CacheProcLayer = parent

    def steps_counter(self, cache: ActsCache):
        return (
            cache.acts.shape[0]
            * torch.ones(
                1,
                dtype=torch.int64,
                device=cache.acts.device,
            )
        ).expand_as(cache.acts[0])

    def get_dead_neurons(self, count_min, threshold):
        return (self.freqs < threshold) & (self._count > count_min)

    def _update_from_cache(self, cache: ActsCache, **kwargs):
        if cache.has.acts:
            self.num_activations += self.activated_counter(cache)
            self._count = self._count + self.steps_counter(cache)

    def activated_counter(self, cache: ActsCache):
        return cache.acts.count_nonzero(dim=0)

    def reset_freqs(self, mask=None, initial_activation=0, initial_count=0):
        if mask is None:
            # assert isinstance(self._count, int)
            self.num_activations = (
                torch.zeros_like(self.num_activations) + initial_activation
            )
            self._count = torch.zeros_like(self._count) + initial_count
        else:
            self.num_activations[mask] = initial_activation
            self._count[mask] = initial_count

    @property
    def freqs(self):
        return self.num_activations / (self._count + 1e-9)


class EMAFreqMixin: ...


AFMC = TypeVar("AFMC", bound=FreqTracker)


class ActFreqCLayer(ComponentLayer):
    activations: AFMC

    def __init__(
        self,
        freq_act_tracker_class: Type[AFMC] = CountingFreqTracker,
        extra_components: List[LayerComponent] = [],
    ):
        super().__init__(
            CacheLayer.from_cfg(cfg),
            [freq_act_tracker_class()] + extra_components,
        )


# afc = ActFreqCLayer(2, CLayerCountingFreqActComponent)
# afc.activations.steps_counter
