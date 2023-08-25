import torch
import torch.nn as nn

from diffusers import UNet2DConditionModel

from type_alias import *


class MyUNet(nn.Module):
    @property
    def inChannel(self) -> int:
        raise NotImplementedError

    @property
    def outChannel(self) -> int:
        raise NotImplementedError

    def forward(
            self, 
            sample          : torch.FloatTensor,
            sigma           : torch.Tensor,
            extract_feature : torch.Tensor | None = None
        ) -> torch.Tensor:

        raise NotImplementedError


class PrecondUNet(UNet2DConditionModel, MyUNet):
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
            sample          : torch.FloatTensor,
            sigma           : torch.Tensor,
            extract_feature : torch.Tensor | None = None
        ) -> torch.Tensor:

        cSkip, cOut, cIn, cNoise = self.GetPrecondSigmas(sigma)
        if cSkip.dim() != 0: cSkip = cSkip[:, None, None, None]
        if cOut .dim() != 0: cOut  = cOut [:, None, None, None]
        if cIn  .dim() != 0: cIn   = cIn  [:, None, None, None]

        modelOut = super().forward(cIn * sample, cNoise, None, class_labels=extract_feature).sample
        return cSkip * sample[:, :3] + cOut * modelOut
