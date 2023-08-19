import torch.nn as nn

from copy import deepcopy

from utils import *


class ModuleEMA(nn.Module):
    def __init__(self, model: nn.Module, beta: float = 0.99, nStepPerUpdate: int = 1):
        super().__init__()
        self.model          = deepcopy(model)
        self.beta           = beta
        self.nStepPerUpdate = nStepPerUpdate
    
        self.__registerModel = [model]
        self.__nUpdateStep   = 0

        self.requires_grad_(False)
        self.eval()
    
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