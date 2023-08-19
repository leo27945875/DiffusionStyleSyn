import torch


def DynamicThreshold(predX0: torch.Tensor, ratio: float = 0.995) -> torch.Tensor:
    s = torch.quantile(predX0.abs().flatten(1), ratio, dim=1)
    s = torch.max(s, torch.tensor(1., device=predX0.device))[:, None, None, None]
    return torch.clip(predX0, -s, s) / s


def RescaleConditionResult(cond: torch.Tensor, scaled: torch.Tensor, phi: float = 0.7) -> torch.Tensor:
    std_cond   = torch.std(cond  , dim=tuple(d for d in range(1, cond  .ndim)), keepdim=True)
    std_scaled = torch.std(scaled, dim=tuple(d for d in range(1, scaled.ndim)), keepdim=True)
    factor = (std_cond / std_scaled) * phi + (1. - phi)
    return scaled * factor