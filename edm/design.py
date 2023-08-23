import torch

from utils      import *
from type_alias import *


class EDM:
    def __init__(
            self,
            nStep      : int   = 100,
            sigmaMin   : float = 0.002,
            sigmaMax   : float = 80.,
            sigmaData  : float = 0.5,
            rho        : float = 7.,
            sampleMode : str   = "log_normal"
        ) -> None:

        assert sampleMode in {"uniform", "log_normal"}, f"[EDM] Invalid sampling mode [{sampleMode}]."

        self.nStep      = nStep
        self.sigmaMin   = sigmaMin
        self.sigmaMax   = sigmaMax
        self.sigmaData  = sigmaData
        self.rho        = rho
        self.sampleMode = sampleMode
        self.offsetStep = 0

    def LossFunc(self, preds: torch.Tensor, trues: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        weights = ((sigmas ** 2 + self.sigmaData ** 2) / (sigmas * self.sigmaData) ** 2)
        return (torch.pow(preds - trues, 2).mean(dim=(1, 2, 3)) * weights).mean()

    def SampleTimes(self, batchSize, pMean=-1.2, pStd=1.2, device="cpu") -> torch.Tensor:
        if self.sampleMode == "log_normal":
            return torch.exp(pMean + pStd * torch.randn((batchSize,), device=device))
        elif self.sampleMode == "uniform":
            return torch.rand((batchSize,)) * (self.sigmaMax - self.sigmaMin) + self.sigmaMin
        else:
            raise AssertionError(f"[EDM] Invalid sampling mode [{self.sampleMode}].")

    def AddNoise(self, images: torch.Tensor, sigmas: torch.Tensor, noises: torch.Tensor | None = None) -> torch.Tensor:
        return images + sigmas[:, None, None, None] * (SampleNoises(images.size(), device=images.device) if noises is None else noises)

    def TimeToSigma(self, time: torch.Tensor | float) -> torch.Tensor | float:
        return time
    
    def SigmaToTime(self, sigma: torch.Tensor | float) -> torch.Tensor | float:
        return sigma
    
    def IndexToTime(self, i: int) -> float:
        if i == self.nStep:
            return 0.
        if self.nStep == 1:
            return self.sigmaMax
        
        invRho = 1. / self.rho
        return ((self.sigmaMax ** invRho) + (i / (self.nStep - 1)) * (self.sigmaMin ** invRho - self.sigmaMax ** invRho)) ** self.rho
    
    def TimeToIndex(self, time: float) -> int:
        if time == 0.:
            return self.nStep

        invRho = 1. / self.rho
        return round((time ** invRho - self.sigmaMax ** invRho) / (self.sigmaMin ** invRho - self.sigmaMax ** invRho) * (self.nStep - 1))
    
    def IndexToSigma(self, i: int) -> float:
        return self.TimeToSigma(self.IndexToTime(i))
    
    def SigmaToIndex(self, sigma: float) -> int:
        return self.TimeToIndex(self.SigmaToTime(sigma))

    def Precondition(self, sigma: torch.Tensor) -> T_Precond:
        sigmaData2 = self.sigmaData ** 2
        inv_sum2   = torch.reciprocal(sigmaData2 + sigma ** 2)

        cSkip  = sigmaData2 * inv_sum2
        cIn    = torch.sqrt(inv_sum2)
        cOut   = sigma * self.sigmaData * cIn
        cNoise = 0.25 * torch.log(sigma + 1e-44)
        return cSkip, cOut, cIn, cNoise