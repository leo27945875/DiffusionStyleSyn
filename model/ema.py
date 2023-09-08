from copy import deepcopy

from .unet import UNet
from utils import *


class ModuleEMA(UNet):
    def __init__(self, model: UNet, beta: float = 0.99, nStepPerUpdate: int = 1):
        super().__init__()
        self.model          = deepcopy(model)
        self.beta           = beta
        self.nStepPerUpdate = nStepPerUpdate
    
        self.__registerModel = [model]
        self.__nUpdateStep   = 0

        self.requires_grad_(False)
        self.eval()
    
    @property
    def inChannel(self) -> int:
        return self.model.inChannel
    
    @property
    def outChannel(self) -> int:
        return self.model.outChannel
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict)
    
    def requires_grad_(self, requires_grad: bool = False):
        return super().requires_grad_(False)

    def train(self, mode: bool = False):
        return super().train(False)
    
    def update(self):
        self.__nUpdateStep += 1
        if self.__nUpdateStep % self.nStepPerUpdate == 0:
            InterpolateParams(self.model, self.__registerModel[0], self.beta)