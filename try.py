import albumentations as A
import numpy as np
import torch
import torch.nn as nn

a = torch.randn(10, 10, 3)
b = a.to("cuda")
print(a.device, b.device)