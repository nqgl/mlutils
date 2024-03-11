from nqgl.mlutils.components.config import WandbDynamicConfig
from nqgl.mlutils.components.cache import Cache, CacheSpec

from nqgl.mlutils.components.component_layer import LayerComponent, ComponentLayer
import torch.nn as nn
from dataclasses import dataclass
import torch
from typing import Optional, Callable, Tuple
from abc import abstractmethod
from torch import Tensor
from jaxtyping import Int, Bool, Float
import torch.nn.functional as F


@dataclass
class ResamplerConfig(WandbDynamicConfig):
    num_to_resample: int = 128
    dead_threshold: float = 3e-6
    min_viable_count: int = 10_000
    check_frequency: int = 100
    norm_encoder_proportional_to_alive: bool = True
    reset_all_freqs_interval: int = 10000
    reset_all_freqs_offset: int = 0
    normalized_encoder_multiplier: float = 0.2


class ResamplingCache(CacheSpec):
    # dead_neurons: torch.Tensor
    saved_x: torch.Tensor
    # resample: Callable
    # saved_y_pred: torch.Tensor


class ResamplerComponent(LayerComponent):
    _default_component_name = "resampler"
    _requires_component = ["activations", "resampling_method"]
    train_cache_watch = []
    eval_cache_watch = []
    resampling_method: "ResamplingMethod"

    def __init__(
        self,
        cfg: ResamplerConfig,
        W_next: Optional[nn.Parameter] = None,
    ):
        self.cfg = cfg
        self.W_next = W_next
        self.T = 0
        self._layer: ComponentLayer = None

    def _register_parent_layer(self, layer: ComponentLayer):
        super()._register_parent_layer(layer)
        self._layer.train_cache_template.register_write_callback(
            "x", self._resample_train_hook_x
        )

    def _resample_train_hook_x(self, cache: ResamplingCache, x):
        self._step()
        if self.is_resample_step():
            # cache.resample = ...
            # cache.resample = lambda x, y_pred, y: self.resample_callback(
            #     cache, x, y_pred, y
            # )
            cache._parent.x = ...
            cache._parent.y = ...
            cache._parent.y_pred = ...
            cache.resample = ...
            cache.resample = lambda **kwargs: self.resample_callback(cache, **kwargs)

    # in the future, can denote this with something like:
    # @ComponentLayer.hook.cache.train.x
    # -> set a class field of any LayerComponent that holds hooks to register
    # and then ComponentLayer is responsible for registering/calling them appropriately
    def is_resample_step(self):
        return self.is_check_step()

    def is_check_step(self):
        return self.T % self.cfg.check_frequency == 0 and self._layer.training

    def _step(self):
        if self._layer.training:
            self.T += 1

    @property
    def freqs(self):
        return self._layer.activations.freqs

    def get_dead_neurons(self):
        return self._layer.activations.get_dead_neurons(
            self.cfg.min_viable_count, self.cfg.dead_threshold
        )

    def reset_activation_frequencies(self, mask=None):
        self._layer.activations.reset_freqs(mask)

    @torch.no_grad()
    def reset_neurons(
        self,
        new_directions: Float[Tensor, "nnz d_in"],
        to_reset: Tensor,  # Int[Tensor] or Boool[Tensor]?
    ):
        if isinstance(to_reset, tuple):
            self.reset_neurons_from_index(new_directions, to_reset)
        elif to_reset.dtype == torch.bool:
            self.reset_neurons_from_mask(new_directions, to_reset)
        else:
            self.reset_neurons_from_index(new_directions, to_reset)

    @torch.no_grad()
    def reset_neurons_from_mask(
        self,
        new_directions: Float[Tensor, "nnz d_in"],
        to_reset: Bool[Tensor, "*#inst d_out"],
    ):  # this may be wrong
        assert new_directions.shape[0] == torch.count_nonzero(to_reset)
        if self.W_next is not None:
            self.W_next.data[to_reset] = self.proc_W_next_directions(new_directions)
        self._layer.cachelayer.W.data.transpose(-2, -1)[to_reset] = (
            self.proc_W_directions(new_directions)
        )
        self._layer.cachelayer.b.data[to_reset] = self.proc_bias_directions(
            new_directions
        )

    @torch.no_grad()
    def reset_neurons_from_index(
        self,
        new_directions: Float[Tensor, "nnz d_out"],
        to_reset: Bool[Tensor, "nnz d_out"],
    ):
        assert (
            isinstance(to_reset, tuple)
            and len(to_reset) == self._layer.cachelayer.W.ndim - 1
        ) or (
            to_reset.ndim == 2
            and to_reset.shape[1] == self._layer.cachelayer.W.ndim - 1
        )

        if to_reset[0].shape[0] == 0:
            return
        if self.W_next is not None:
            self.W_next.data[to_reset] = self.proc_W_next_directions(new_directions)
        self._layer.cachelayer.W.data.transpose(-2, -1)[to_reset] = (
            self.proc_W_directions(new_directions)
        )
        self._layer.cachelayer.b.data[to_reset] = self.proc_bias_directions(
            new_directions
        )

    def proc_W_next_directions(self, new_directions: Float[Tensor, "nnz d_out"]):
        return F.normalize(
            new_directions, dim=-1
        )  # just need to call norm dec after this

    def get_dead_neurons_for_norm(self):
        return self.get_dead_neurons()

    def proc_W_directions(self, new_directions: Float[Tensor, "nnz d_in"]):
        dead = self.get_dead_neurons_for_norm()
        if torch.all(dead):
            print("warning: all neurons dead")
            return F.normalize(new_directions, dim=-1)
        alives = self._layer.cachelayer.W.transpose(-2, -1)[~dead]
        return (
            F.normalize(new_directions, dim=-1)
            * alives.norm(dim=-1).mean()
            * self.cfg.normalized_encoder_multiplier
        )

    def proc_bias_directions(self, new_directions: Float[Tensor, "nnz d_out"]):
        return 0

    @abstractmethod
    def resample_callback(self, cache, x=None, y_pred=None, y=None): ...

    def _update_from_cache(self, cache: ResamplingCache, **kwargs):
        # training = kwargs.get("training", False)
        # if training:
        #     self._step()
        if (
            self.T - self.cfg.reset_all_freqs_offset
        ) % self.cfg.reset_all_freqs_interval == 0:
            self.reset_activation_frequencies()

        if self.is_check_step():
            self.check_dead()
        # if self.is_resample_step():

    def get_neurons_to_resample(self):
        return self.get_dead_neurons()

    def check_dead(self):
        pass


class ResamplingMethod(ResamplerComponent):
    # _default_component_name = "resampling_method"
    # _requires_component = ["resampler"]

    def __init__(self, cfg: ResamplerConfig, W_next: Optional[nn.Parameter] = None):
        super().__init__(cfg=cfg, W_next=W_next)
        self.cumulative_num_resampled = 0

    # @abstractmethod
    @torch.inference_mode()
    def resample_callback(self, cache: ResamplingCache, x=None, y_pred=None, y=None):
        x = x if x is not None else cache._parent.x
        y = y if y is not None else cache._parent.y
        y_pred = y_pred if y_pred is not None else cache._parent.y_pred
        dead = self.get_neurons_to_resample()
        directions = self._get_directions(cache, x, y_pred, y)
        if dead is None or directions is None:
            cache.num_resampled = 0
            return
        if dead.dtype == torch.bool:
            to_reset = dead.nonzero()
        else:
            to_reset = dead
        to_reset = to_reset[: min(directions.shape[0], self.cfg.num_to_resample)]

        directions = directions[: to_reset.shape[0]]
        cache.num_resampled = to_reset.shape[0]
        self.cumulative_num_resampled += to_reset.shape[0]
        cache.cumulative_num_resampled = self.cumulative_num_resampled
        to_reset = to_reset.unbind(-1)
        self.reset_neurons(directions, to_reset)

    def reset_neurons(self, new_directions: Tensor, to_reset: Tensor):
        super().reset_neurons(new_directions, to_reset)
        mask = self.get_dead_neurons()
        mask[:] = False
        mask[to_reset] = True
        self.reset_activation_frequencies(mask)

    def _get_directions(self, cache: ResamplingCache, x, y_pred, y):
        return self.get_directions(cache, x, y_pred, y)

    @abstractmethod
    def get_directions(self, cache, x, y_pred, y): ...

    # def resample_from_new_directions(self, ):


class GeneratedBatchResampler(ResamplingMethod):
    def __init__(
        self,
        cfg: ResamplerConfig,
        buffer,
        forward_model,
        W_next: Optional[nn.Parameter] = None,
    ):
        super().__init__(cfg=cfg, W_next=W_next)
        self.buffer = buffer
        self.forward_model = forward_model

    def _get_directions(self, cache, x, y_pred, y):
        y_l = []
        y_pred_l = []
        x_l = []
        with torch.inference_mode():
            for i in range(...):
                x = self.buffer.next()
                if isinstance(x, tuple):
                    x, y = x
                else:
                    y = x
                y_pred_l.append(self.forward_model(x))
                y_l.append(y)
                x_l.append(x)
        y = torch.cat(y_l)
        y_pred = torch.cat(y_pred_l)
        x = torch.cat(x_l)
        return self.get_directions(cache, x, y_pred, y)
        ...


@dataclass
class QueuedResamplerConfig(ResamplerConfig):
    resample_frequency: int = 100
    resampling_cycle: Tuple[int, int] = 1, 1
    append_to_queue: bool = True


class QueuedResampler(ResamplingMethod):
    def __init__(
        self, cfg: QueuedResamplerConfig, W_next: Optional[nn.Parameter] = None
    ):
        super().__init__(cfg=cfg, W_next=W_next)
        self.queued = None

    def is_resample_step(self):
        return (
            self.T % self.cfg.resample_frequency == 0
            and self._layer.training
            and self.queued is not None
            and self.queued.shape[0] > 0
        )

    def get_dead_neurons_for_norm(self):
        if self.queued is None:
            return self.get_dead_neurons()
        mask = self.get_dead_neurons()
        mask[:] = False
        mask[self.queued] = True
        return mask

    def check_dead(self):
        super().check_dead()
        self.queued = (
            torch.unique(
                torch.cat((self.queued, self.get_dead_neurons().nonzero())),
                sorted=False,
                dim=0,
            )
            if self.queued is not None and self.cfg.append_to_queue
            else self.get_dead_neurons().nonzero()
        )

    def _update_from_cache(self, cache: ResamplingCache, **kwargs):
        cache.num_queued_for_reset = (
            self.queued.shape[0] if self.queued is not None else 0
        )
        return super()._update_from_cache(cache, **kwargs)

    def get_neurons_to_resample(self):
        if (
            self.queued is not None
            and self.T % self.cfg.resampling_cycle[1]
            > self.cfg.resampling_cycle[1] - self.cfg.resampling_cycle[0]
        ):
            q = self.queued[: self.cfg.num_to_resample]
            self.queued = self.queued[self.cfg.num_to_resample :]
            return q
        return None


@dataclass
class TopKResamplingConfig:
    resample_top_k: int = None


class TopKResampling(ResamplingMethod):
    def get_directions(self, cache, x, y_pred, y):
        ranking = self.get_ranking_metric(cache, x, y_pred, y)
        k = self.cfg.resample_top_k or self.cfg.num_to_resample
        indices = torch.topk(ranking, k, largest=True).indices
        return self.get_directions_for_indices(cache, x, y_pred, y, indices)

    # @abstractmethod
    def get_ranking_metric(self, cache, x, y_pred, y):
        return (y - y_pred).pow(2).mean(dim=-1)

    def get_directions_for_indices(self, cache, x, y_pred, y, indices):
        x = x[indices]
        y_pred = y_pred[indices]
        y = y[indices]
        return super().get_directions(cache, x, y_pred, y)


class RandomResamplingDirections(ResamplingMethod):
    def get_directions(self, cache, x, y_pred, y):
        return torch.randn(self.cfg.num_to_resample, x.shape[-1], device=x.device)


class DiffResamplingDirections(ResamplingMethod):
    def get_directions(self, cache, x, y_pred, y):
        return y - y_pred


class YResamplingDirections(ResamplingMethod):
    def get_directions(self, cache, x, y_pred, y):
        return y


class SVDResampling(ResamplingMethod):
    def get_directions(self, cache, x, y_pred, y):
        print("x", x.shape)
        print("y_pred", y_pred.shape)
        print("y", y.shape)
        u, s, v = torch.svd(y - y_pred)
        print("u", u.shape)
        print("s", s.shape)
        print("v", v.shape)
        # k = (self.cfg.num_to_resample + 1) // 2
        k = self.cfg.num_to_resample
        sort = torch.argsort(s.abs(), descending=True)
        # best_k_directions = v[sort[:k]] * s[sort[:k]].unsqueeze(-1)
        print("v", v.shape)
        best_k_directions = v[sort[:k]] * s[sort[:k]].unsqueeze(-1)
        print("bkd", best_k_directions.shape)
        # perf = (best_k_directions @ (y - y_pred).transpose(-2, -1)).mean(
        #     dim=-1
        # )  # (k batch).mean(batch)
        perf = ((y - y_pred) @ best_k_directions.transpose(-2, -1)).pow(3).mean(dim=0)
        print("perf", perf.shape)
        dirs = best_k_directions * torch.sign(perf).unsqueeze(-1)
        return dirs

        return torch.cat((best_k_directions, -best_k_directions), dim=0)
        signs = torch.sign(perf)
        return dirs * signs.unsqueeze(-1)
