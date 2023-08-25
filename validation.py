import torch
from torch.utils.data  import DataLoader
from torchvision.utils import make_grid, save_image

from edm        import EDMSampler
from model      import *
from utils      import *
from type_alias import *


@torch.inference_mode()
def Valid(
        sampler      : EDMSampler,
        dataloader   : DataLoader,
        denoiser     : ModuleEMA | MyUNet,
        extractor    : Extractor,
        device       : torch.device,
        saveFilename : str
):
    isDenosierEMA       = isinstance(denoiser, ModuleEMA)
    isDenoiserTraining  = denoiser .training
    isExtractorTraining = extractor.training
    denoiser .eval()
    extractor.eval()

    generateds, nTotal = None, 0
    for images, masks, toExtracts in dataloader:

        images, masks, toExtracts = images.to(device), masks.to(device), toExtracts.to(device)
        
        B, C, H, W = images.size()
        nTotal += B

        denoiseArgs = {
            "cond": {
                "concat" : masks,
                "extract": extractor(toExtracts)
            },
            "uncond": {
                "concat" : torch.zeros([B, (denoiser.model.inChannel if isDenosierEMA else denoiser.inChannel) - C, H, W], device=device),
                "extract": extractor.MakeUncondTensor(B, device=device)
            }
        }
        batchRes   = sampler.Run(denoiser, B, None, denoiseArgs)
        generateds = DefaultConcatTensor(generateds, batchRes)
    
    save_image(
        make_grid(generateds.clamp(0., 1.), nrow=round(nTotal ** 0.5)), 
        saveFilename
    )

    if isDenoiserTraining: 
        denoiser.train()
    if isExtractorTraining: 
        extractor.train()
    
    torch.cuda.empty_cache()