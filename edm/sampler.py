import torch
from torchvision.utils import make_grid, save_image

from tqdm import trange

from model    import MyUNet
from .rescale import DynamicThreshold, RescaleConditionResult
from .design  import EDM


class EDMSampler:
    def __init__(
            self,
            diffusion    : EDM,
            imageSize    : tuple[int, int]     = (128, 128),
            sChurn       : float               = 40., 
            sNoise       : float               = 1.003, 
            st           : tuple[float, float] = (0., float("inf")), 
            isStochastic : bool                = True,
            isThreshold  : bool                = False,
            device       : torch.device        = torch.device("cpu"),
        ):

        self.diffusion    = diffusion
        self.nStep        = diffusion.nStep
        self.sigmaMin     = diffusion.sigmaMin
        self.sigmaMax     = diffusion.sigmaMax
        self.rho          = diffusion.rho

        self.imageSize    = imageSize
        self.st           = st
        self.sChurn       = sChurn
        self.sNoise       = sNoise
        self.device       = device
        self.isStochastic = isStochastic
        self.isThreshold  = isThreshold
        self.gamma        = min(sChurn / self.nStep, 2. ** 0.5 - 1.)
    
    @torch.inference_mode()
    def Run(
            self, 
            denoiser            : MyUNet, 
            batchSize           : int, 
            saveFilename        : str  = None, 
            denoiseArgs         : dict = {}, 
            isReturnDenormImage : bool = True
        ) -> torch.Tensor:

        t1  = self.diffusion.IndexToSigma(0)
        img = self.SampleNoises(batchSize, std=t1)
        for i in trange(self.nStep):
            t0 = t1
            gamma = self.GetGamma(t0)
            if self.isStochastic and gamma != 0.:
                th = (1 + gamma) * t0
                x0 = img + self.SampleNoises(batchSize, std=self.sNoise) * (th ** 2 - t0 ** 2) ** 0.5
            else:
                th = t0
                x0 = img
            
            d0 = (x0 - self.Denoise(denoiser, x0, th, denoiseArgs)) / th
            t1 = self.diffusion.IndexToSigma(i + 1)
            x1 = x0 + (t1 - th) * d0
            if i != self.nStep - 1:
                d1 = (x1 - self.Denoise(denoiser, x1, t1, denoiseArgs)) / t1
                x1 = x0 + (t1 - th) * (d0 + d1) * 0.5
            
            img = x1
            
        imgDenorm = (img + 1.) * 0.5
        if saveFilename:
            save_image(
                make_grid(imgDenorm.clamp(0., 1.), nrow=round(batchSize ** 0.5)), 
                saveFilename
            )

        return imgDenorm if isReturnDenormImage else img

    def Denoise(self, denoiser: MyUNet, x: torch.Tensor, sigma: float, denoiseArgs: dict = {}) -> torch.Tensor:
        if self.isThreshold:
            return DynamicThreshold(denoiser(x, torch.tensor(sigma, device=self.device)))
        
        return denoiser(x, torch.tensor(sigma, device=self.device))
    
    def GetGamma(self, sigma: float) -> float:
        return self.gamma if self.st[0] <= sigma <= self.st[1] else 0.
    
    def SampleNoises(self, batchSize: int, mean: float = 0., std: float = 1.):
        return torch.randn((batchSize, 3, *self.imageSize), device=self.device) * std + mean


class EDMCondSampler(EDMSampler):
    def __init__(
            self,
            diffusion    : EDM,
            imageSize    : tuple[int, int]     = (128, 128),
            sChurn       : float               = 40., 
            sNoise       : float               = 1.003, 
            st           : tuple[float, float] = (0., float("inf")),
            isStochastic : bool                = True,
            isThreshold  : bool                = False,
            wCond        : float               = 1.,
            rescalePhi   : float | None        = None,
            device       : torch.device        = torch.device("cpu"),
        ) -> None:
        super().__init__(diffusion, imageSize, sChurn, sNoise, st, isStochastic, isThreshold, device)
        self.wCond      = wCond
        self.rescalePhi = rescalePhi

    def Denoise(self, denoiser: MyUNet, x: torch.Tensor, sigma: float, denoiseArgs: dict) -> torch.Tensor:

        xCond   = torch.cat([x, denoiseArgs["cond"  ]["concat"]], dim=1)
        xUncond = torch.cat([x, denoiseArgs["uncond"]["concat"]], dim=1)

        sigmas = torch.tensor(sigma, device=self.device)

        condOuts   = denoiser(xCond  , sigmas, denoiseArgs["cond"  ]["extract"])
        uncondOuts = denoiser(xUncond, sigmas, denoiseArgs["uncond"]["extract"])

        outs = (1. + self.wCond)  * condOuts - self.wCond * uncondOuts
        if self.rescalePhi:
            outs = RescaleConditionResult(condOuts, outs, self.rescalePhi)
        if self.isThreshold: 
            outs = DynamicThreshold(outs)

        return outs
