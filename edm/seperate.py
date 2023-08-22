import copy
from typing import Any

from .design import EDM


def Seperate(diffusion: EDM, nSeperate: int, index: int, otherArgs: dict[str, Any] = {}) -> EDM:
    """
    Seperate a partition of EDM. It's used for training eDiffi.
    """
    nStep = diffusion.nStep
    inter = nStep / nSeperate
    
    startStep = index * inter
    endStep   = startStep + inter

    startStep, endStep = round(startStep), round(endStep)
    sigmaMax = diffusion.IndexToSigma(startStep)
    sigmaMin = diffusion.IndexToSigma(endStep)

    newDiffusion            = copy.deepcopy(diffusion)
    newDiffusion.nStep      = endStep - startStep
    newDiffusion.offsetStep = startStep
    newDiffusion.sigmaMax   = sigmaMax
    newDiffusion.sigmaMin   = sigmaMin
    for attr, value in otherArgs.items():
        setattr(newDiffusion, attr, value)

    return newDiffusion