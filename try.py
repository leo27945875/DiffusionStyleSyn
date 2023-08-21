import torch
import torch.nn.functional as F

a = torch.randn(8, 256)
zq = F.interpolate(a, size=(32, 32), mode="nearest")
print(zq.shape)