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
    set_dec_to_enc: bool = False
    add_to_max_acts: float = None
    max_undying: int = None


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

    def is_check_step(self):
        return (
            self.dead is False
            or super().is_check_step()
            and self.T > self.cfg.wait_to_check_dead
        )

    def is_resample_step(self):
        return True

    @torch.no_grad()
    def resample_callback(self, cache: ResamplingCache, x=None, y_pred=None, y=None):
        inactive_dead = self.dead & (cache.acts == 0).all(dim=0)
        self._layer.cachelayer.b.grad[inactive_dead] = 0
        self._layer.cachelayer.b[inactive_dead] *= self.cfg.bias_decay
        cache.num_undying = ...
        cache.num_undying = self.dead.count_nonzero() if self.dead is not False else 0
        cache.num_dead = ...
        cache.num_dead = self.get_dead_neurons().count_nonzero()
        if self.cfg.set_dec_to_enc:
            self.W_next.transpose(-2, -1)[self.dead] = (
                self._layer.cachelayer.W.transpose(-2, -1)[self.dead]
            )

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
            new_dead = self.get_dead_neurons()
            if self.cfg.max_undying is not None:
                new_dead = new_dead & (
                    torch.rand_like(self._layer.activations.freqs)
                    < min(
                        1,
                        (
                            (self.cfg.max_undying - still_dead.count_nonzero())
                            / (
                                self.cfg.max_undying
                                + new_dead.count_nonzero()
                                - still_dead.count_nonzero()
                            )
                        ).item(),
                    )
                )
            self.dead = new_dead | still_dead
            # self.reset_activation_frequencies(undied)

    def nonlinearity(self, x):
        if self.dead is False:
            return torch.relu(x)
        out = torch.where(self.dead, self.cfg.undying_relu(x), torch.relu(x))
        if self.cfg.add_to_max_acts:
            z = torch.zeros_like(out)
            i = x[:, self.dead].argmax(dim=0)
            z[:, self.dead][i, torch.arange(len(i))] = self.cfg.add_to_max_acts
            out = out + z
        return out
