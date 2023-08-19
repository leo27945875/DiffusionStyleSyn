import torch

from typing import Callable


T_Precond      = tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float, torch.Tensor]
T_Precond_Func = Callable[[torch.Tensor], T_Precond]
