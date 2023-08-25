import torch
import torch.nn as nn

from typing import Callable

from .unet import MyUNet
from .ema  import ModuleEMA
from edm   import EDM
from utils import LoadCheckpoint


class Ensembler(MyUNet):
    def __init__(
            self, 
            models            : list[MyUNet], 
            diffusion         : EDM, 
            trainingIdx       : int | None = None, 
            isSaveMode        : bool       = False, 
            isTestOnlineModel : bool       = False, 
            isShowMessage     : bool       = False
        ):
        super().__init__()
        self.models            = nn.ModuleList(models)
        self.diffusion         = diffusion
        self.isSaveMode        = isSaveMode
        self.isTestOnlineModel = isTestOnlineModel
        self.isShowMessage     = isShowMessage
        self.interval          = diffusion.nStep / len(models)

        self.__inChannel, self.__outChannel = self.__CheckConsistancy(models)

        self.__lastModelIdx : int | None          = None
        self.__trainingIdx  : int | None          = trainingIdx
        self.__onlineModel  : list[MyUNet] | None = []          # Using list to avoid being registered to nn.Module.

        
        if isSaveMode:
            self.cpu()
    
    @classmethod
    def InitFromFiles(
        cls, 
        ensembleFiles     : list[str], 
        BuildModelFunc    : Callable[[], MyUNet], 
        diffusion         : EDM, 
        trainingIdx       : int | None = None, 
        isSaveMode        : bool       = False, 
        isTestOnlineModel : bool       = False, 
        isShowMessage     : bool       = False
    ) -> "Ensembler":
        
        ensembleList, onlineModel = [], None
        for i, file in enumerate(ensembleFiles):
            denoiser = BuildModelFunc()
            if file is not None:
                LoadCheckpoint(file, ema=denoiser, isOnlyLoadWeight=True)
            if i == trainingIdx:
                onlineModel = denoiser
                denoiser    = ModuleEMA(onlineModel)

            ensembleList.append(denoiser)
        
        ensembler = cls(ensembleList, diffusion, trainingIdx, isSaveMode, isTestOnlineModel, isShowMessage)
        ensembler.SetOnlineModel(onlineModel)
        ensembler.requires_grad_(False)
        ensembler.eval()
        return ensembler
    
    def __getitem__(self, i) -> MyUNet:
        return self.models[i]
    
    @property
    def inChannel(self) -> int:
        return self.__inChannel

    @property
    def outChannel(self) -> int:
        return self.__outChannel
    
    @property
    def onlineModel(self) -> MyUNet:
        if not self.__onlineModel:
            return None
        
        return self.__onlineModel[0]
    
    @property
    def offlineModel(self) -> ModuleEMA | None:
        if self.__trainingIdx is None:
            return None
        
        return self.models[self.__trainingIdx]
    
    def forward(
            self, 
            sample          : torch.FloatTensor, 
            sigma           : torch.Tensor, 
            extract_feature : torch.Tensor | None = None
        ) -> torch.Tensor:
        
        model, modelIdx = self.__ChooseModel(sigma)

        if (
            self.isTestOnlineModel and 
            self.__onlineModel     and 
            modelIdx == self.__trainingIdx
        ):
            model = self.onlineModel
            model.eval()
        
        isTraining = model.training

        if self.isSaveMode and (modelIdx != self.__lastModelIdx):
            if self.__lastModelIdx is not None:
                self.models[self.__lastModelIdx].cpu()

            model.cuda()
            if self.isShowMessage:
                print(f"\n[Ensembler] Switched to model[{modelIdx}]")

        self.__lastModelIdx = modelIdx

        output = model(sample, sigma, extract_feature)
        if isTraining: 
            model.train()

        return output

    def to(self, device: str | torch.device):
        if self.isSaveMode:
            return self
        
        return super().to(device)
    
    def cuda(self, device: int | torch.device | None = None):
        if self.isSaveMode:
            return self
        
        return super().cuda(device)
    
    def SetOnlineModel(self, model: MyUNet) -> None:
        assert model.inChannel  == self.__inChannel , f"[Ensembler.SetOnlineModel()] Found model.inChannel is not consistant."
        assert model.outChannel == self.__outChannel, f"[Ensembler.SetOnlineModel()] Found model.outChannel is not consistant."
        self.__onlineModel = [model]
    
    def __CheckConsistancy(self, models: list[MyUNet]) -> tuple[int, int]:
        inChannel, outChannel = models[0].inChannel, models[0].outChannel
        for model in models[1:]:
            assert model.inChannel  == inChannel , f"[Ensembler.__CheckConsistancy()] Found model.inChannel is not consistant."
            assert model.outChannel == outChannel, f"[Ensembler.__CheckConsistancy()] Found model.outChannel is not consistant."
        
        return inChannel, outChannel

    def __ChooseModel(self, sigma: torch.Tensor) -> tuple[nn.Module, int]:
        if sigma.dim() != 0:
            s = sigma[0].item()
        else:
            s = sigma.item()

        modelIdx = int(self.diffusion.SigmaToIndex(s) / self.interval)
        return self.models[modelIdx], modelIdx