import torch
import torch.nn as nn

from typing import Callable

from .unet import MyUNet
from .ema  import ModuleEMA
from edm   import EDM
from utils import LoadCheckpoint


class Ensembler(MyUNet):
    def __init__(self, models: list[MyUNet], diffusion: EDM, isSaveMode: bool = False, trainingIdx: int | None = None):
        super().__init__()
        self.models     = nn.ModuleList(models)
        self.diffusion  = diffusion
        self.isSaveMode = isSaveMode
        self.interval   = diffusion.nStep / len(models)

        self.__inChannel, self.__outChannel = self.__CheckConsistancy(models)

        self.__trainingIdx : int | None             = trainingIdx
        self.__testingEMA  : list[ModuleEMA] | None = None          # Using list to avoid being registered to nn.Module.

        self.__lastModelIdx : int | None = None
        
        if isSaveMode:
            self.cpu()
    
    @classmethod
    def InitFromFiles(cls, ensembleFiles: list[str], BuildModelFunc: Callable[[], MyUNet], diffusion: EDM, isSaveMode: bool = False) -> "Ensembler":
        ensembleList, trainingIdx = [], None
        for i, file in enumerate(ensembleFiles):
            denoiser = BuildModelFunc()
            if file is not None:
                LoadCheckpoint(file, ema=ModuleEMA(denoiser), isOnlyLoadWeight=True)
            else:
                assert trainingIdx is None, "[Ensembler] Only allow one training model in the Ensembler for now."
                trainingIdx = i

            ensembleList.append(denoiser)
        
        ensembler = cls(ensembleList, diffusion, isSaveMode)
        ensembler.SetTrainingIndex(trainingIdx)
        return ensembler
    
    @property
    def inChannel(self) -> int:
        return self.__inChannel

    @property
    def outChannel(self) -> int:
        return self.__outChannel
    
    def __getitem__(self, i) -> MyUNet:
        return self.models[i]
    
    def forward(
            self, 
            sample          : torch.FloatTensor, 
            sigma           : torch.Tensor, 
            extract_feature : torch.Tensor | None = None
        ) -> torch.Tensor:
        
        model, modelIdx = self.__ChooseModel(sigma)

        if self.__testingEMA and modelIdx == self.__trainingIdx:
            model = self.__testingEMA[0]
        
        if self.isSaveMode:
            isSwitchDevice = modelIdx != self.__lastModelIdx
            if isSwitchDevice: model.cuda()
            out = model(sample, sigma, extract_feature)
            if isSwitchDevice: model.cpu()
            return out
        
        self.__lastModelIdx = modelIdx
        return model(sample, sigma, extract_feature)

    def to(self, device: str | torch.device):
        if self.isSaveMode:
            return self
        
        return super().to(device)
    
    def cuda(self, device: int | torch.device | None = None):
        if self.isSaveMode:
            return self
        
        return super().cuda(device)
    
    def SetTrainingIndex(self, trainingIdx: int) -> None:
        self.__trainingIdx = trainingIdx
    
    def SetTestingEMA(self, ema: ModuleEMA) -> None:
        self.__testingEMA = [ema]
    
    def __CheckConsistancy(self, models: list[MyUNet]) -> tuple[int, int]:
        inChannel, outChannel = models[0].inChannel, models[0].outChannel
        for model in models[1:]:
            assert model.inChannel  == inChannel , f"[Ensembler] Found model.inChannel is not consistant."
            assert model.outChannel == outChannel, f"[Ensembler] Found model.outChannel is not consistant."
        
        return inChannel, outChannel

    def __ChooseModel(self, sigma: torch.Tensor) -> tuple[nn.Module, int]:
        if sigma.dim() != 0:
            s = sigma[0].item()
        else:
            s = sigma.item()

        modelIdx = int(self.diffusion.SigmaToIndex(s) / self.interval)
        return self.models[modelIdx], modelIdx