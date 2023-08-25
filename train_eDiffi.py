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
from validation import Valid


def BuildModel(PreconditionFunc: T_Precond_Func, nClass: int, baseChannel: int, attnChannel: int, extractorOutChannel: int):
    """
    Build diffusion model arch in CPU.
    """
    return PrecondUNet(
        GetPrecondSigmas                      = PreconditionFunc,
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
        projection_class_embeddings_input_dim = extractorOutChannel
    )


def Train(
        seed             : int         = 0,
        nEpoch           : int         = 500,
        batchSize        : int         = 160,
        gradAccum        : int         = 8,
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
        extractorName    : str         = "ViT-B/32",
        nClass           : int         = 150,
        ckptFile         : str | None  = None,
        isOnlyLoadWeight : bool        = False,
        isValidFirst     : bool        = False,
        isValidEMA       : bool        = True,
        isCompile        : bool        = False,
        isFixExtractor   : bool        = True,
        isUseFixFeature  : bool        = True,    
        dataFolder       : str         = "data",
        saveFolder       : str         = "save",
        visualFolder     : str         = "visual",

        # Ensemble args:
        nSeperate     : int       = 2,
        seperateIdx   : int       = 0,
        seperateArgs  : dict      = {"sampleMode": "uniform"},
        ensembleFiles : list[str] = ["save/EDM_64/EDM_Epoch1000.pth", "save/EDM_64/EDM_Epoch1000.pth"],
        isSaveGPUMode : bool      = True
):
    # Random seed:
    SeedEverything(seed)

    # File & Folder:
    modelName    = f"eDiff-i[{seperateIdx}]"
    saveFolder   = f"{saveFolder}/{modelName}"
    visualFolder = f"{visualFolder}/{modelName}"
    saveCkptName = f"{saveFolder}/{modelName}.pth"
    
    os.makedirs(saveFolder  , exist_ok=True)
    os.makedirs(visualFolder, exist_ok=True)


    # Device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Diffusion:
    wholeDiffusion = EDM(nStep)
    if seperateArgs:
        trainDiffusion = Seperate(wholeDiffusion, nSeperate, seperateIdx, seperateArgs)

    # Sampler:
    sampler = EDMCondSampler(wholeDiffusion, (imageSize, imageSize), device=device)

    # Extractor:
    if isUseFixFeature:
        extractor = ExtractorPlaceholder(backbone=extractorName)
    else:
        extractor = VisualExtractor(backbone=extractorName)

    # Ensemble:
    assert ensembleFiles[seperateIdx] is None, f"[ensembleFiles[seperateIdx]] must be None."
    assert all([(file is not None) for i, file in enumerate(ensembleFiles) if i != seperateIdx]), f"Files correspond to auxiliary model must not be None."
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

    ensembler.to(device)
    extractor.to(device)

    # Data:
    trainset, validset = MakeDatasets(
        dataFolder         = dataFolder,
        imageSize          = imageSize,
        extractorTransform = extractor.GetPreprocess(isFromNormalized=True), 
        isUseFixFeature    = isUseFixFeature
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
        Valid(
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
            Valid(
                sampler      = sampler,
                dataloader   = validloader,
                denoiser     = ensembler,
                extractor    = extractor,
                device       = device, 
                saveFilename = f"{visualFolder}/{modelName}_Epoch{epoch}.png"
            )
    
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


if __name__ == '__main__':

    Train()