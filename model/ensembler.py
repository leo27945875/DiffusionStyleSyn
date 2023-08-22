import torch
import torch.nn as nn

from type_alias import torch

from .unet import PrecondUNet, MyUNet
from edm   import EDM


class Ensembler(MyUNet):
    def __init__(self, models: list[PrecondUNet], diffusion: EDM, isSaveMode: bool = False):
        super().__init__()
        self.models     = nn.ModuleList(models)
        self.diffusion  = diffusion
        self.isSaveMode = isSaveMode
        self.interval   = diffusion.nStep / len(models)

        self.__inChannel, self.__outChannel = self.__CheckConsistancy(models)

        self.__lastModelIdx = None
    
    @property
    def inChannel(self) -> int:
        return self.__inChannel

    @property
    def outChannel(self) -> int:
        return self.__outChannel
    
    def forward(
            self, 
            sample          : torch.FloatTensor, 
            sigma           : torch.Tensor, 
            extract_feature : torch.Tensor
        ) -> torch.Tensor:
        
        model, modelIdx = self.__ChooseModel(sigma)

        isSwitchDevice = modelIdx != self.__lastModelIdx
        self.__lastModelIdx = modelIdx

        if self.isSaveMode:
            if isSwitchDevice: model.cuda()
            out = model(sample, sigma, extract_feature)
            if isSwitchDevice: model.cpu()
            return out
        
        return model(sample, sigma, extract_feature)

    def to(self, device: str | torch.device):
        if self.isSaveMode:
            return self
        
        return super().to(device)
    
    def cuda(self, device: int | torch.device | None = None):
        if self.isSaveMode:
            return self
        
        return super().cuda(device)
    
    def __CheckConsistancy(self, models: list[PrecondUNet]) -> None:
        inChannel, outChannel = models[0].inChannel, models[0].outChannel
        for model in models[1:]:
            assert model.inChannel  == inChannel , f"[Ensembler] Found model.inChannel is not consistant."
            assert model.outChannel == outChannel, f"[Ensembler] Found model.outChannel is not consistant."
        
        return inChannel, outChannel

    def __ChooseModel(self, sigma: torch.Tensor) -> tuple[nn.Module, int]:
        modelIdx = int(self.diffusion.SigmaToIndex(sigma.item()) / self.interval)
        return self.models[modelIdx], modelIdx