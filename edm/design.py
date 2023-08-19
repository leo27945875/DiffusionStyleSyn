import torch

from utils      import *
from type_alias import *


class EDM:
    def __init__(
            self,
            timeRange : int   = 100,
            sigmaMin  : float = 0.002,
            sigmaMax  : float = 80.,
            sigmaData : float = 0.5,
            rho       : float = 7.
        ) -> None:
        self.timeRange = timeRange
        self.sigmaMin  = sigmaMin
        self.sigmaMax  = sigmaMax
        self.sigmaData = sigmaData
        self.rho       = rho

    def LossFunc(self, preds: torch.Tensor, trues: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        weights = ((sigmas ** 2 + self.sigmaData ** 2) / (sigmas * self.sigmaData) ** 2)
        return (torch.pow(preds - trues, 2).mean(dim=(1, 2, 3)) * weights).mean()

    def SampleTimes(self, batchSize, pMean=-1.2, pStd=1.2, device="cpu") -> torch.Tensor:
        return torch.exp(pMean + pStd * torch.randn((batchSize,), device=device))

    def AddNoise(self, images: torch.Tensor, sigmas: torch.Tensor, noises: torch.Tensor | None = None) -> torch.Tensor:
        return images + sigmas[:, None, None, None] * (SampleNoises(images.size(), device=images.device) if noises is None else noises)

    def IndexToTime(self, i: int) -> float:
        if i == self.timeRange:
            return 0.
        
        if self.timeRange == 1:
            return self.sigmaMax
        
        invRho = 1. / self.rho
        return ((self.sigmaMax ** invRho) + (i / (self.timeRange - 1)) * (self.sigmaMin ** invRho - self.sigmaMax ** invRho)) ** self.rho
    
    def IndexToSigma(self, i: int) -> float:
        return self.TimeToSigma(self.IndexToTime(i))
    
    def TimeToSigma(self, times: torch.Tensor | float) -> torch.Tensor | float:
        return times

    def Precondition(self, sigma: torch.Tensor) -> T_Precond:
        sigmaData2 = self.sigmaData ** 2
        inv_sum2   = torch.reciprocal(sigmaData2 + sigma ** 2)

        cSkip  = sigmaData2 * inv_sum2
        cIn    = torch.sqrt(inv_sum2)
        cOut   = sigma * self.sigmaData * cIn
        cNoise = 0.25 * torch.log(sigma + 1e-44)
        return cSkip[:, None, None, None], cOut[:, None, None, None], cIn[:, None, None, None], cNoise