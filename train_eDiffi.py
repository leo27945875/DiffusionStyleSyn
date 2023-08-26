import torch
from torch.optim       import RAdam
from torch.utils.data  import DataLoader
from lion_pytorch      import Lion

import os
import random
from functools import partial

from data       import MakeDatasets
from edm        import EDM, EDMCondSampler, Seperate
from model      import *
from utils      import *
from type_alias import *
from validation import Valid, ModelBackToCPU


def Train(
        seed             : int         = 0,
        nEpoch           : int         = 300,
        batchSize        : int         = 180,
        gradAccum        : int         = 4,
        lr               : float       = 2e-5,
        nWorker          : int         = 8,
        validFreq        : int         = 5,
        ckptFreq         : int         = 1,
        isAmp            : bool        = True,
        pUncond          : float       = 0.1,
        nStep            : int         = 100,
        imageSize        : tuple       = 64,
        baseChannel      : int         = 192,
        attnChannel      : int         = 16,
        nClass           : int         = 150,
        ckptFile         : str | None  = None,
        isOnlyLoadWeight : bool        = False,
        isValidFirst     : bool        = False,
        isValidEMA       : bool        = True,
        isCompile        : bool        = False,
        isFixExtractor   : bool        = True,
        dataFolder       : str         = "data_80",
        saveFolder       : str         = "save",
        visualFolder     : str         = "visual",
        fixedFeatureFile : str | None  = "ADE20K-outdoor_CLIP.pth",
        featureAxisNum   : int         = 3,
        modelName        : str         = "eDiff-i",

        # Ensemble args:
        nSeperate     : int       = 2,
        seperateIdx   : int       = 0,
        seperateArgs  : dict      = {"sampleMode": "uniform"},
        ensembleFiles : list[str] = ["save/eDiff-i.pth", "save/eDiff-i.pth"],
        isSaveGPUMode : bool      = False
):
    # Random seed:
    SeedEverything(seed)

    # File & Folder:
    modelName    = f"{modelName}_{imageSize}[{seperateIdx}]"
    saveFolder   = f"{saveFolder}/{modelName}"
    visualFolder = f"{visualFolder}/{modelName}"
    saveCkptName = f"{saveFolder}/{modelName}.pth"
    
    os.makedirs(saveFolder  , exist_ok=True)
    os.makedirs(visualFolder, exist_ok=True)

    # Validation:
    ValidFunc = Valid if isSaveGPUMode else ModelBackToCPU(Valid)

    # Device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Diffusion:
    wholeDiffusion = EDM(nStep)
    if seperateArgs:
        trainDiffusion = Seperate(wholeDiffusion, nSeperate, seperateIdx, seperateArgs)

    # Sampler:
    sampler = EDMCondSampler(wholeDiffusion, (imageSize, imageSize), device=device)

    # Extractor:
    assert featureAxisNum in {2, 3}, f"[Train] The parameter [featureAxisNum] must be 2 or 3. But got {featureAxisNum} instead."
    if fixedFeatureFile:
        match featureAxisNum:
            case 2: extractor = ExtractorPlaceholder("clip")
            case 3: extractor = ExtractorPlaceholder("vqgan")
    else:
        match featureAxisNum:
            case 2: extractor = CLIPImageEncoder()
            case 3: extractor = VQGAN()

    # Ensemble:
    BuildModelFunc = partial(
        BuildModel, PreconditionFunc=wholeDiffusion.Precondition, nClass=nClass, baseChannel=baseChannel, attnChannel=attnChannel, extractorOutChannel=extractor.outChannel
    )
    ensembler = Ensembler.InitFromFiles(ensembleFiles, BuildModelFunc, wholeDiffusion, seperateIdx, isSaveGPUMode, not isValidEMA)
    model     = ensembler.onlineModel
    ema       = ensembler.offlineModel

    optimizer = Lion(model.parameters(), lr=lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=isAmp)

    if isCompile:
        torch.compile(extractor)
        torch.compile(model)
        torch.compile(ensembler)

    if isFixExtractor:
        extractor.requires_grad_(False)
        extractor.eval()

    ensembler.cpu()
    extractor.to(device)
    model    .to(device)

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
        resumeEpoch = LoadCheckpoint(ckptFile, model, extractor, ema, optimizer, None, scaler, None, isOnlyLoadWeight)
    else:
        resumeEpoch = 0
    
    # Training:
    if isValidFirst:
        ValidFunc(
            sampler      = sampler,
            dataloader   = validloader,
            denoiser     = ensembler,
            extractor    = extractor,
            device       = device, 
            saveFilename = f"{visualFolder}/{modelName}_Valid_Check.png"
        )

    for epoch in range(resumeEpoch + 1, nEpoch + 1):

        losses = Metric()

        for batch, (images, masks, toExtracts) in enumerate(trainloader, 1):

            if random.random() < pUncond:
                images, masks, toExtracts = images, None, None
            
            loss = GetLoss(model, extractor, trainDiffusion, images, masks, toExtracts, gradAccum, isAmp, device)
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
            SaveCheckpoint(epoch, saveCkptName, model, extractor, ema, optimizer, None, scaler)

        # Validation:
        if epoch % validFreq == 0:
            ValidFunc(
                sampler      = sampler,
                dataloader   = validloader,
                denoiser     = ensembler,
                extractor    = extractor,
                device       = device, 
                saveFilename = f"{visualFolder}/{modelName}_Epoch{epoch}.png"
            )
    
    del ensembler, model, ema, extractor, optimizer
    torch.cuda.empty_cache()
    return saveCkptName


def GetLoss(
        denoiser   : MyUNet,
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


def Main():

    LEVEL             = 1
    N_TRAINING_EPOCHS = [300, 300]
    INIT_WEIGHT_CKPTS = ["save/EDM_64/EDM_Epoch1000.pth"]
    CHECK_POINT_FILES = []

    ################################## Training Pipeline ##################################

    assert LEVEL >= 1, "[Main] [LEVEL] must >= 1."

    nSeperate   = 2 ** (LEVEL)
    nPretrained = 2 ** (LEVEL - 1)

    assert len(INIT_WEIGHT_CKPTS) == nPretrained, f"[Main] Length of [INIT_WEIGHT_CKPTS] is not enough. (Must be equal to {nPretrained})"
    assert len(N_TRAINING_EPOCHS) == nSeperate  , f"[Main] Length of [N_TRAINING_EPOCHS] must be equal to 2 ** [LEVEL]"
    assert len(CHECK_POINT_FILES) <= nSeperate  , f"[Main] Length of [CHECK_POINT_FILES] must be less than 2 ** [LEVEL]."

    ensembleFiles = [
        INIT_WEIGHT_CKPTS[i // 2] for i in range(nSeperate)
    ]

    print("\n" + "=" * 50 + " Start training eDiff-i " + "=" * 50)

    for i in range(nSeperate):
        print(f"\n\nEnsemble init ckpt : [{ensembleFiles}]\n")
        print(f"Training ensemble [{i}] ... ")

        nEpoch   = N_TRAINING_EPOCHS[i]
        ckptFile = CHECK_POINT_FILES[i] if i < len(CHECK_POINT_FILES) else None
        if ckptFile is not None:
            print(f"Got checkoint file : [{ckptFile}].")
            if nEpoch != 0:
                ensembleFiles[i] = ckptFile
        
        if nEpoch:
            ensembleFiles[i] = Train(
                nEpoch        = nEpoch,
                ckptFile      = ckptFile,
                nSeperate     = nSeperate,
                seperateIdx   = i,
                ensembleFiles = ensembleFiles,
                modelName     = f"eDiff-i_L{LEVEL}"
            )
        else:
            print("No training epoch, so skip training process.")
    
    print("\n\nFinish training. Ckpt files :")
    for i, file in enumerate(ensembleFiles):
        print(f"Ensemble [{i}] : [{file}]")

    print("\n")


if __name__ == '__main__':

    Main()