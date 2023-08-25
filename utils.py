import torch

import os
import os.path as osp
import cv2
import json
import math
import random
import numpy as np

PI = math.pi


class Metric:
    def __init__(self):
        self.val = 0.
        self.num = 0
    
    def Record(self, x, n=1) -> None:
        if np.isnan(x) or np.isinf(x):
            return

        self.val += x
        self.num += n
    
    def Mean(self) -> float:
        return self.val / self.num


def SeedEverything(seed, isFixCudnn=False) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if isFixCudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def LoadJSON(filename: str) -> dict | list:
    with open(filename, "r") as f:
        return json.load(f)


def SaveCheckpoint(epoch, filename, model, extractor, ema=None, optimizer=None, scheduler=None, scaler=None, targetNet=None) -> None:
    ckpt = {
        "epoch"     : epoch,
        "model"     : model    .state_dict(),
        "extractor" : extractor.state_dict() if extractor else None,
        "ema"       : ema      .state_dict() if ema       else None,
        "optimizer" : optimizer.state_dict() if optimizer else None,
        "scheduler" : scheduler.state_dict() if scheduler else None,
        "scaler"    : scaler   .state_dict() if scaler    else None,
        "targetNet" : targetNet.state_dict() if targetNet else None
    }
    torch.save(ckpt, filename)


def LoadCheckpoint(filename, model=None, extractor=None, ema=None, optimizer=None, scheduler=None, scaler=None, targetNet=None, isOnlyLoadWeight=False) -> int:
    ckpt = torch.load(filename, map_location="cpu")

    if model and ckpt["model"]:
        model.load_state_dict(ckpt["model"])

    if extractor and ckpt["extractor"]:
        extractor.load_state_dict(ckpt["extractor"])

    if ema and ckpt["ema"]:
        ema.load_state_dict(ckpt["ema"])

    if not isOnlyLoadWeight:
        if optimizer and ckpt["optimizer"]: optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt["scheduler"]: scheduler.load_state_dict(ckpt["scheduler"])
        if scaler    and ckpt["scaler"   ]: scaler   .load_state_dict(ckpt["scaler"   ])
        if targetNet and ckpt["targetNet"]: targetNet.load_state_dict(ckpt["targetNet"])

    return ckpt["epoch"]


def ReadRGBImage(filename: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def GetBasename(filepath: str, isTrimExt: bool = False, newExt: str | None = None) -> str:
    if isTrimExt:
        return osp.splitext(osp.basename(filepath))[0]
    if newExt:
        return osp.splitext(osp.basename(filepath))[0] + "." + newExt

    return osp.basename(filepath)


def ChangeFolder(filename: str, newFolder: str, newExt: str | None = None) -> str:
    basename = GetBasename(filename, newExt=newExt)
    return f"{newFolder}/{basename}"


def PrintShape(a):
    print(a.shape)
    return a


def SampleNoises(size, device="cpu"):
    return torch.randn(size, device=device)


def InterpolateParams(model0: torch.nn.Module, model1: torch.nn.Module, ratio: float) -> None:
    for (name0, param0), (name1, param1) in zip(model0.named_parameters(), model1.named_parameters()):
        assert name0 == name1, "[InterpolateParams] The name of two parameters are not consistant."
        param0.data.lerp_(param1.data.to(param0.device), 1. - ratio)


def MaskToOnehot(mask: torch.Tensor, nClass: int):
    h, w, device = mask.shape[0], mask.shape[1], mask.device
    onehot = torch.zeros([nClass, h, w], device=device)
    onehot.scatter_(0, mask.unsqueeze(0), torch.ones([1, h, w], device=device))
    return onehot


def DefaultConcatTensor(t0: torch.Tensor | None, t1: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if t0 is None:
        return t1

    return torch.cat([t0, t1], dim=dim)


def FloatToBatch(batchSize: int, f: float, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    return torch.tensor([f], dtype=torch.float, device=device).expand(batchSize)