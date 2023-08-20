import torch
import torch.nn as nn

import clip

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Extractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @property
    def inChannel(self) -> int:
        raise NotImplementedError

    @property
    def outChannel(self) -> int:
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def GetPreprocess(self, isFromNormalized: bool = False) -> A.Compose | None:
        raise NotImplementedError
    
    def MakeUncondTensor(self, batchSize: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        raise NotImplementedError
    

class ExtractorPlaceholder(Extractor):
    def __init__(self, backbone: str = "ViT-B/32"):
        super().__init__()
        net = clip.load(backbone)[0].visual
        self.__inChannel  = net.conv1.weight.size(1)
        self.__outChannel = net.output_dim

    @property
    def inChannel(self) -> int:
        return self.__inChannel
    
    @property
    def outChannel(self) -> int:
        return self.__outChannel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def GetPreprocess(self, isFromNormalized: bool = False) -> A.Compose:
        return None

    def MakeUncondTensor(self, batchSize: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.zeros([1, self.outChannel], device=device).expand(batchSize, -1)
    

class VisualExtractor(Extractor):
    def __init__(self, backbone: str = "ViT-B/32"):
        super().__init__()
        self.net = clip.load(backbone)[0].visual
    
    @property
    def inChannel(self) -> int:
        return self.net.conv1.weight.size(1)

    @property
    def outChannel(self) -> int:
        return self.net.output_dim
    
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