import torch
from torch.optim       import RAdam
from torch.utils.data  import DataLoader
from lion_pytorch      import Lion

import random

from data       import MakeDatasets
from edm        import EDM, EDMCondSampler
from model      import *
from utils      import *
from type_alias import *
from validation import Valid, ModelBackToCPU


def Train(
        seed             : int         = 0,
        nEpoch           : int         = 1000,
        batchSize        : int         = 224,
        gradAccum        : int         = 32,
        lr               : float       = 2e-5,
        nWorker          : int         = 8,
        validFreq        : int         = 5,
        ckptFreq         : int         = 1,
        isAmp            : bool        = True,
        pUncond          : float       = 0.1,
        nStep            : int         = 100,
        imageSize        : tuple       = 128,
        baseChannel      : int         = 256,
        attnChannel      : int         = 8,
        nClass           : int         = 150,
        ckptFile         : str | None  = "save/EDM_128/EDM_128.pth",
        isOnlyLoadWeight : bool        = False,
        isValidFirst     : bool        = True,
        isValidEMA       : bool        = True,
        isCompile        : bool        = False,
        isFixExtractor   : bool        = True,
        dataFolder       : str         = "data",
        saveFolder       : str         = "save",
        visualFolder     : str         = "visual",
        fixedFeatureFile : str | None  = "ADE20K-outdoor_CLIP.pth",
        featureAxisNum   : int         = 2,
        modelName        : str         = "EDM"
):
    # Random seed:
    SeedEverything(seed)

    # File & Folder:
    modelName    = f"{modelName}_{imageSize}"
    saveFolder   = f"{saveFolder}/{modelName}"
    visualFolder = f"{visualFolder}/{modelName}"
    saveCkptName = f"{saveFolder}/{modelName}.pth"
    
    os.makedirs(saveFolder  , exist_ok=True)
    os.makedirs(visualFolder, exist_ok=True)

    # Validation:
    ValidFunc = ModelBackToCPU(Valid) if isValidEMA else Valid

    # Device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Diffusion:
    diffusion = EDM(nStep)

    # Sampler:
    sampler = EDMCondSampler(diffusion, (imageSize, imageSize), device=device)

    # Model:
    assert featureAxisNum in {2, 3}, f"[Train] The parameter [featureAxisNum] must be 2 or 3. But got {featureAxisNum} instead."
    if fixedFeatureFile:
        match featureAxisNum:
            case 2: extractor = ExtractorPlaceholder("clip")
            case 3: extractor = ExtractorPlaceholder("vqgan")
    else:
        match featureAxisNum:
            case 2: extractor = CLIPImageEncoder()
            case 3: extractor = VQGAN()
            
    denoiser  = BuildModel(diffusion.Precondition, nClass, baseChannel, attnChannel, extractor.outChannel, extractor.crossAttnChannel)
    optimizer = Lion(denoiser.parameters(), lr=lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=isAmp)
    ema       = ModuleEMA(denoiser)

    if isCompile:
        torch.compile(extractor)
        torch.compile(denoiser)
        torch.compile(ema)

    if isFixExtractor:
        extractor.requires_grad_(False)
        extractor.eval()
    
    denoiser.to(device)
    extractor.to(device)
    ema.cpu()

    # Data:
    trainset, validset = MakeDatasets(
        dataFolder         = dataFolder,
        imageSize          = imageSize,
        extractorTransform = extractor.GetPreprocess(isFromNormalized=True), 
        fixedFeatureFile   = fixedFeatureFile
    )
    trainloader = DataLoader(trainset, batchSize // gradAccum, True, pin_memory=True, num_workers=nWorker)
    validloader = DataLoader(validset, len(validset), False, pin_memory=True)

    # Load checkpoint:
    if ckptFile:
        resumeEpoch = LoadCheckpoint(ckptFile, denoiser, extractor, ema, optimizer, None, scaler, None, isOnlyLoadWeight)
    else:
        resumeEpoch = 0
    
    # Training:
    if isValidFirst:
        ValidFunc(
            sampler      = sampler,
            dataloader   = validloader,
            denoiser     = ema if isValidEMA else denoiser,
            extractor    = extractor,
            device       = device, 
            saveFilename = f"./visual/{modelName}_Valid_Check.png"
        )

    for epoch in range(resumeEpoch + 1, nEpoch + 1):

        losses = Metric()

        for batch, (images, masks, toExtracts) in enumerate(trainloader, 1):

            if random.random() < pUncond:
                images, masks, toExtracts = images, None, None
            
            loss = GetLoss(denoiser, extractor, diffusion, images, masks, toExtracts, gradAccum, isAmp, device)
            scaler.scale(loss).backward()
            if batch % gradAccum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()
            
            losses.Record(loss.item())
            try:
                print(f"\r| Epoch {epoch} | Batch {batch} | Loss {losses.Mean() :.6f}", end="")
            except ZeroDivisionError:
                print(f"\r| Epoch {epoch} | Batch {batch} | Loss (Error)", end="")
        
        print("")

        # Checkpoint:
        if epoch % ckptFreq == 0:
            SaveCheckpoint(epoch, saveCkptName, denoiser, extractor, ema, optimizer, None, scaler)

        # Validation:
        if epoch % validFreq == 0:
            ValidFunc(
                sampler      = sampler,
                dataloader   = validloader,
                denoiser     = ema if isValidEMA else denoiser,
                extractor    = extractor,
                device       = device, 
                saveFilename = f"./visual/{modelName}_Epoch{epoch}.png"
            )


def GetLoss(
        denoiser   : UNet,
        extractor  : Extractor,
        diffusion  : EDM,
        images     : torch.Tensor,
        masks      : torch.Tensor | None,
        toExtracts : torch.Tensor | None,
        gradAccum  : int,
        isAmp      : bool,
        device     : torch.device
) -> torch.Tensor:

    B, C, H, W = images.size()

    images = images.to(device)
    if masks      is not None: masks      = masks     .to(device)
    if toExtracts is not None: toExtracts = toExtracts.to(device)
    
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