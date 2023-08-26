import torch
import torch.nn as nn

import clip
from taming.models.vqgan import VQModel

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2
from omegaconf import OmegaConf

from utils import *


class Extractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @property
    def inChannel(self) -> int:
        """
        The input channel of the first conv layer.
        """
        raise NotImplementedError

    @property
    def outChannel(self) -> int:
        """
        The output channel of the last conv layer.
        """
        raise NotImplementedError
    
    @property
    def crossAttnChannel(self) -> int | None:
        """
        The cross-attention channel for which the diffusion model need to prepare.
        """
        raise NotImplementedError
    
    @property
    def latentAxisNum(self) -> int:
        """
        The output dimension of latent code without the dimension of batchsize.
        """
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Be used to get the latent code of extractor (or encoder).
        """
        raise NotImplementedError
    
    def GetPreprocess(self, isFromNormalized: bool = False) -> A.Compose | None:
        raise NotImplementedError
    
    def MakeUncondTensor(self, batchSize: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        raise NotImplementedError
    

class ExtractorPlaceholder(Extractor):
    def __init__(self, mode, **kwargs):
        super().__init__()
        assert mode in {"clip", "vqgan"}, f"[ExtractorPlaceholder] Invalid network mode [{mode}]."
        match mode:
            case "clip" : net = CLIPImageEncoder(kwargs.get("backbone", "ViT-B/32"))
            case "vqgan": net = VQGAN(
                kwargs.get("configFile", "model/logs/vqgan_imagenet_f16_1024/configs/model.yaml"),
                kwargs.get("ckptFile"  , "model/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt")
            )

        self.__inChannel        = net.inChannel
        self.__outChannel       = net.outChannel
        self.__crossAttnChannel = net.crossAttnChannel
        self.__latentDim        = net.latentAxisNum
        del net

    @property
    def inChannel(self) -> int:
        return self.__inChannel
    
    @property
    def outChannel(self) -> int:
        return self.__outChannel
    
    @property
    def crossAttnChannel(self) -> int | None:
        return self.__crossAttnChannel
    
    @property
    def latentAxisNum(self) -> int:
        return self.__latentDim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def GetPreprocess(self, isFromNormalized: bool = False) -> A.Compose:
        return None

    def MakeUncondTensor(self, batchSize: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.zeros([1, self.outChannel], device=device).expand(batchSize, -1)
    

class CLIPImageEncoder(Extractor):
    def __init__(self, backbone: str = "ViT-B/32"):
        super().__init__()
        self.net = clip.load(backbone)[0].visual.float()
    
    @property
    def inChannel(self) -> int:
        return self.net.conv1.weight.size(1)

    @property
    def outChannel(self) -> int:
        return self.net.output_dim
    
    @property
    def crossAttnChannel(self) -> int | None:
        return None
    
    @property
    def latentAxisNum(self) -> int:
        return 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
        --------------------------
        x: (B, 3, 224, 224)

        Outputs:
        --------------------------
        out: (B, 1024)
        """
        return self.net(x)
    
    def GetPreprocess(self, isFromNormalized: bool = False):

        size = self.net.input_resolution

        def ShortSideResize(img: np.ndarray, *args, **kwargs) -> np.ndarray:
            nonlocal size
            H, W = img.shape[0], img.shape[1]
            if H > W:
                H = round(H * (size / W))
                W = size
            else:
                W = round(W * (size / H))
                H = size
            
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        

        if isFromNormalized:
            return A.Compose([
                A.Lambda(image=lambda img, *args, **kwargs: (img + 1.) * 0.5),
                A.Lambda(image=ShortSideResize),
                A.CenterCrop(size, size),
                A.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), max_pixel_value=1.)
            ])

        return A.Compose([
            A.Lambda(image=ShortSideResize),
            A.CenterCrop(size, size),
            A.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), max_pixel_value=255.),
            ToTensorV2()
        ])
    
    def MakeUncondTensor(self, batchSize: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.zeros([1, self.outChannel], device=device).expand(batchSize, -1)
    

class VQGAN(VQModel, Extractor):
    def __init__(
        self, 
        configFile : str | None = "model/logs/vqgan_imagenet_f16_1024/configs/model.yaml", 
        ckptFile   : str | None = "model/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt"
    ):
        super().__init__(**OmegaConf.load(configFile).model.params)
        if ckptFile:
            self.load_state_dict(
                FilterStateDict(torch.load(ckptFile, map_location="cpu")["state_dict"], excludePrefix="loss")
            )

    def EncodeNotQuantizedLatent(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def DecodeNotQuantizedLatent(self, h: torch.Tensor, isForceQuantize: bool = True) -> torch.Tensor:
        if isForceQuantize:
            quant, _, _ = self.quantize(h)
        else:
            quant = h

        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    @property
    def inChannel(self) -> int:
        return self.encoder.in_channels
    
    @property
    def outChannel(self) -> int:
        return self.quant_conv.weight.size(0)

    @property
    def crossAttnChannel(self) -> int | None:
        return self.quant_conv.weight.size(0)
    
    @property
    def latentAxisNum(self) -> int:
        return 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method will call [self.EncodeNotQuantizedLatent(x)].

        Inputs:
        --------------------------
        x: (B, 3, 256, 256)

        Outputs:
        --------------------------
        out: (B, 256, 16, 16)
        """
        return self.EncodeNotQuantizedLatent(x)
    
    def GetPreprocess(self, isFromNormalized: bool = False):

        size = 256

        def ShortSideResize(img: np.ndarray, *args, **kwargs) -> np.ndarray:
            nonlocal size
            H, W = img.shape[0], img.shape[1]
            if H > W:
                H = round(H * (size / W))
                W = size
            else:
                W = round(W * (size / H))
                H = size
            
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        

        if isFromNormalized:
            return A.Compose([
                A.Lambda(image=ShortSideResize),
                A.CenterCrop(size, size),
            ])

        return A.Compose([
            A.Lambda(image=ShortSideResize),
            A.CenterCrop(size, size),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), max_pixel_value=255.),
            ToTensorV2()
        ])
    
    def MakeUncondTensor(self, batchSize: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.zeros([1, self.outChannel, 1], device=device).expand(batchSize, -1)