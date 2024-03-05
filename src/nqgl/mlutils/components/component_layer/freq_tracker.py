from nqgl.mlutils.components.cache_layer import ActsCache, CacheProcLayer, CacheLayer
from nqgl.mlutils.components.component_layer import (
    LayerComponent,
    ComponentLayer,
)

import torch
from abc import ABC, abstractmethod
from typing import List, Type, TypeVar


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

    @classmethod
    def bind_init_args(cls, **kwargs):
        return lambda parent: cls(parent=parent, **kwargs)


class CLayerCountingFreqActComponent(FreqMonitorComponent):
    train_cache_watch = ["acts"]
    eval_cache_watch = []

    def __init__(self):
        self.num_activations = 0
        self._count = 0
        # self._parent = None
        # self.parent: CacheProcLayer = parent

    def steps_counter(self, cache: ActsCache):
        return cache.acts.shape[0]

    def get_dead_neurons(self, count_min, threshold):
        return super().get_dead_neurons(count_min, threshold)

    def _update_from_cache(self, cache: ActsCache):
        if cache.has.acts:
            self.num_activations += self.activated_counter(cache)
            self._count += self.steps_counter(cache)

    def activated_counter(self, cache: ActsCache):
        return cache.acts.count_nonzero(dim=0)

    def reset_freqs(self, mask=None, initial_activation=0, initial_count=0):
        if mask is None:
            assert isinstance(self.count, int)
            self.active = torch.zeros_like(self.active) + initial_activation
            self.count = initial_count
        else:
            self.active[mask] = initial_activation
            self.count[mask] = initial_count

    @property
    def freqs(self):
        return self.num_activations / (self._count + 1e-9)


class EMAFreqMixin: ...


AFMC = TypeVar("AFMC", bound=FreqMonitorComponent)


class ActFreqCLayer(ComponentLayer):
    activations: AFMC

    def __init__(
        self,
        cfg,
        freq_act_tracker_class: Type[AFMC] = CLayerCountingFreqActComponent,
        extra_components: List[LayerComponent] = [],
    ):
        super().__init__(
            cfg,
            CacheLayer.from_cfg(cfg),
            [freq_act_tracker_class()] + extra_components,
        )


# afc = ActFreqCLayer(2, CLayerCountingFreqActComponent)
# afc.activations.steps_counter
