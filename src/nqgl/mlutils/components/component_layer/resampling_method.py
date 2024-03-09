from nqgl.mlutils.components.component_layer.resampler import ResamplerComponent


import torch


class QueuedResettingComponent(ResamplerComponent):
    re_init_queued: callable
    ...

    # was re_init_neurons
    @torch.no_grad()
    def resample_from_x_diff(self, x_diff): ...

    # def re_init_neurons
    # -> rename "resample"

    # def resampling_check

    # def reset_neurons

    # def reset_activation_frequencies
    #     -> reset_freqs

    # def get_activation_frequencies(self):
    #     -> freqs property

    # def resample

    # def resample_from_cache(self, cache):
    # self.resample_from_x_diff(cache.x - cache.x_pred)


class GhostGradResettingLinear: ...
