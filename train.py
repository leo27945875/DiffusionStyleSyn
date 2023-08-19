import torch
from torch.optim       import RAdam
from torch.utils.data  import DataLoader
from torchvision.utils import make_grid, save_image

import random

from data  import MakeDatasets
from edm   import EDM, EDMCondSampler
from model import *
from utils import *


def Train(
        seed             : int        = 0,
        nEpoch           : int        = 1000,
        batchSize        : int        = 256,
        gradAccum        : int        = 16,
        lr               : float      = 1e-4,
        nWorker          : int        = 4,
        validFreq        : int        = 10,
        ckptFreq         : int        = 10,
        isAmp            : bool       = True,
        pUncond          : float      = 0.1, 
        nSteps           : int        = 100,
        imageSize        : tuple      = 128,
        baseChannel      : int        = 128,
        attnChannel      : int        = 64,
        extractorName    : str        = "RN50",
        nClass           : int        = 150,
        ckptFile         : str        = None,
        isValidFirst     : bool       = True,
        isCompile        : bool       = False,
        isFixExtractor   : bool       = True,
        fixedFeatureFile : str | None = "data/feature/ADE20K-outdoor_Features.pth"
):
    # Random seed:
    SeedEverything(seed)

    # Device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Diffusion:
    diffusion = EDM(nSteps)

    # Sampler:
    sampler = EDMCondSampler(diffusion, (imageSize, imageSize), device=device)

    # Model:
    if fixedFeatureFile:
        extractor = ExtractorPlaceholder(backbone=extractorName).to(device)
    else:
        extractor = VisualExtractor(backbone=extractorName).to(device)

    denoiser = PrecondUNet(
        GetPrecondSigmas                      = diffusion.Precondition,
        in_channels                           = 3 + nClass,
        out_channels                          = 3,
        block_out_channels                    = (baseChannel, baseChannel * 2, baseChannel * 3, baseChannel * 4),
        down_block_types                      = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types                        = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim                    = attnChannel,
        resnet_time_scale_shift               = "ada_group",          # "default", "scale_shift", "ada_group", "spatial"
        class_embed_type                      = "simple_projection",
        class_embeddings_concat               = True,
        cross_attention_dim                   = (baseChannel, baseChannel * 2, baseChannel * 3, baseChannel * 4),
        projection_class_embeddings_input_dim = extractor.outChannel
    ).to(device)

    optimizer = RAdam(denoiser.parameters(), lr=lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=isAmp)
    ema       = ModuleEMA(denoiser, beta=0.9999, nStepPerUpdate=1).to(device)

    if isCompile:
        torch.compile(extractor)
        torch.compile(denoiser)
        torch.compile(ema)

    if isFixExtractor:
        extractor.requires_grad_(False)
        extractor.eval()

    # Data:
    trainset, validset = MakeDatasets(
        extractorTransform = extractor.GetPreprocess(isFromNormalized=True), 
        fixedFeatureFile   = fixedFeatureFile
    )
    trainloader = DataLoader(trainset, batchSize // gradAccum, True, pin_memory=True, num_workers=nWorker)
    validloader = DataLoader(validset, len(validset), False, pin_memory=True)

    # Load checkpoint:
    if ckptFile:
        resumeEpoch = LoadCheckpoint(ckptFile, denoiser, extractor, ema, optimizer, None, scaler)
    else:
        resumeEpoch = 0
    
    # Training:
    if isValidFirst:
        Valid(
            sampler      = sampler,
            dataloader   = validloader,
            denoiser     = ema,
            extractor    = extractor,
            device       = device, 
            saveFilename = f"./visual/EDM_Valid_Check.png"
        )

    for epoch in range(resumeEpoch + 1, nEpoch + 1):

        losses = Metric()

        for batch, (images, masks, toExtracts) in enumerate(trainloader, 1):

            if random.random() < pUncond:
                images, masks, toExtracts = images.to(device), None, None
            else:
                images, masks, toExtracts = images.to(device), masks.to(device), toExtracts.to(device)
            
            loss = GetLoss(denoiser, extractor, diffusion, images, masks, toExtracts, gradAccum, isAmp)
            scaler.scale(loss).backward()
            if batch % gradAccum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()
            
            losses.Record(loss.item())
            print(f"\r| Epoch {epoch} | Batch {batch} | Loss {losses.Mean() :.6f}", end="")
        
        print("")

        # Checkpoint:
        if epoch % ckptFreq == 0:
            SaveCheckpoint(epoch, f"./save/EDM_Epoch{epoch}.pth", denoiser, ema, optimizer, None, scaler)

        # Validation:
        if epoch % validFreq == 0:
            Valid(
                sampler      = sampler,
                dataloader   = validloader,
                denoiser     = ema,
                extractor    = extractor,
                device       = device, 
                saveFilename = f"./visual/EDM_Epoch{epoch}.png"
            )
            

@torch.inference_mode()
def Valid(
        sampler      : EDMCondSampler,
        dataloader   : DataLoader,
        denoiser     : ModuleEMA,
        extractor    : Extractor,
        device       : torch.device,
        saveFilename : str
):
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
                "concat" : torch.zeros([B, denoiser.model.inChannel - C, H, W], device=device),
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


def GetLoss(
        denoiser   : PrecondUNet,
        extractor  : Extractor,
        diffusion  : EDM,
        images     : torch.Tensor,
        masks      : torch.Tensor | None,
        toExtracts : torch.Tensor | None,
        gradAccum  : int,
        isAmp      : bool
) -> torch.Tensor:

    device     = images.device
    B, C, H, W = images.size()
    
    with torch.cuda.amp.autocast(enabled=isAmp):

        sigmas = diffusion.TimeToSigma(diffusion.SampleTimes(B, device=device))

        if masks is None:
            masks = torch.zeros([B, denoiser.inChannel - C, H, W], device=device)

        if toExtracts is None:
            style = extractor.MakeUncondTensor(B, device)
        else:
            style = extractor(toExtracts)
        
        x = diffusion.AddNoise(images, sigmas)
        x = torch.cat([x, masks], dim=1)

        loss = diffusion.LossFunc(
            denoiser(x, sigmas, style), images, sigmas
        )
        
    return loss / gradAccum


if __name__ == '__main__':

    Train()