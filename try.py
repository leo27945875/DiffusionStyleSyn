import albumentations as A
import numpy as np
import torch
import torch.nn as nn

a = np.random.randn(224, 224, 3)
m = nn.Conv2d(0,0,0)