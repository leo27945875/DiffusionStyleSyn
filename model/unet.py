import torch

from diffusers import UNet2DConditionModel

from typing import Union

from type_alias import *


class PrecondUNet(UNet2DConditionModel):
    def __init__(self, GetPrecondSigmas: T_Precond_Func, **super_kwargs):
        super().__init__(**super_kwargs)
        self.GetPrecondSigmas = GetPrecondSigmas
    
    @property
    def inChannel(self) -> int:
        return self.conv_in.weight.size(1)

    @property
    def outChannel(self) -> int:
        return self.conv_out.weight.size(0)

    def forward(
            self, 
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            extract_feature: torch.Tensor
        ) -> torch.Tensor:

        cSkip, cOut, cIn, cNoise = self.GetPrecondSigmas(timestep)
        modelOut = super().forward(cIn * sample, cNoise, None, class_labels=extract_feature).sample
        return cSkip * sample + cOut * modelOut