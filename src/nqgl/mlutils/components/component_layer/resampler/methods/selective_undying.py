from torch.nn.parameter import Parameter
from nqgl.mlutils.components.component_layer.resampler import (
    ResamplingMethod,
    QueuedResampler,
    ResamplerConfig,
    ResamplerComponent,
)
from nqgl.mlutils.components.component_layer.resampler.resampler import ResamplingCache
from nqgl.mlutils.components.nonlinearities.undying import undying_relu
from dataclasses import dataclass
import torch
from nqgl.mlutils.components.nonlinearities.serializable import SerializableNonlinearity


@dataclass
class SelectiveUndyingResamplerConfig(ResamplerConfig):
    undying_relu: SerializableNonlinearity = SerializableNonlinearity(
        "undying_relu",
        {
            "k": 1,
            "l": 0.01,
            "l_mid_neg": 0.002,
            "l_low_pos": 0.005,
            "l_low_neg": 0.002,
        },
    )
    bias_decay: float = 0.9999
    alive_thresh_mul: float = 2
    resample_before_step: bool = True
    wait_to_check_dead: int = 0


class SelectiveUndyingResampler(ResamplerComponent):
    cfg: SelectiveUndyingResamplerConfig

    def __init__(
        self,
        cfg: SelectiveUndyingResamplerConfig,
        W_next: Parameter | None = None,
        get_optim_fn=None,
    ):
        super().__init__(cfg, W_next, get_optim_fn)
        self.dead = False

    def is_resample_step(self):
        return True

    @torch.no_grad()
    def resample_callback(self, cache: ResamplingCache, x=None, y_pred=None, y=None):
        inactive_dead = self.dead & (cache.acts == 0).all(dim=0)
        self._layer.cachelayer.b.grad[inactive_dead] = 0
        self._layer.cachelayer.b[inactive_dead] *= self.cfg.bias_decay
        cache.num_dead = ...
        cache.num_dead = self.dead.count_nonzero() if self.dead is not False else 0

    def check_dead(self):
        if self.dead is False:
            self.dead = self.get_dead_neurons()
        else:
            still_dead = self.dead & self._layer.activations.get_dead_neurons(
                self.cfg.min_viable_count,
                self.cfg.dead_threshold * self.cfg.alive_thresh_mul,
            )
            undied = self.dead & ~still_dead
            self.reset_adam(undied, dead=self.dead)
            self.dead = self.get_dead_neurons() | still_dead
            # self.reset_activation_frequencies(undied)

    def nonlinearity(self, x):
        if self.dead is False:
            return torch.relu(x)
        return torch.where(self.dead, self.cfg.undying_relu(x), torch.relu(x))
