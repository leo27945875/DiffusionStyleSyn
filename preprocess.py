import torch

import os
import cv2
import glob
from tqdm import tqdm

from utils import *
from model import Extractor, CLIPImageEncoder, VQGAN


def ExtractAndSaveVisualFeatures(
        extractor    : str,
        imageFolder  : str = "data_80/image",
        saveFolder   : str = "data_80/feature",
        saveFilename : str = "ADE20K-outdoor_VQGAN.pth"
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    extractor: Extractor = eval(extractor)().to(device)
    extractor.requires_grad_(False)
    extractor.eval()

    preprocess = extractor.GetPreprocess()

    os.makedirs(saveFolder, exist_ok=True)

    with torch.inference_mode():
        filenames, features = [], None
        for imageFile in tqdm(glob.glob(f"{imageFolder}/*")):
            image: torch.Tensor
            image = preprocess(image=ReadRGBImage(imageFile))["image"]
            image = image.unsqueeze(0).to(device)
            features = DefaultConcatTensor(features, extractor(image).cpu())
            filenames.append(GetBasename(imageFile, isTrimExt=True))
    
    print(f"Saving features (shape = {features.size()}) ... ", end="")
    torch.save({
        "filenames": filenames,
        "features" : features
    }, f"{saveFolder}/{saveFilename}")
    print("Done !\n")

    print("-" * 50 + " [Filenames] " + "-" * 50)
    for filename in filenames:
        print(filename)


def ResizeDataset(srcFolder: str = "data", targetSize: int = 80):

    dstFolder = f"data_{targetSize}"

    imageSrcFolder, maskSrcFolder = f"{srcFolder}/image", f"{srcFolder}/mask"
    imageDstFolder, maskDstFolder = f"{dstFolder}/image", f"{dstFolder}/mask"

    os.makedirs(imageDstFolder, exist_ok=True)
    os.makedirs(maskDstFolder , exist_ok=True)

    srcImageFiles, srcMaskFiles = glob.glob(f"{imageSrcFolder}/*.jpg"), glob.glob(f"{maskSrcFolder}/*.png")
    for srcImageFile, srcMaskFile in zip(srcImageFiles, srcMaskFiles):
        srcImage, srcMask = cv2.imread(srcImageFile), cv2.imread(srcMaskFile, cv2.IMREAD_GRAYSCALE)
        assert srcImage.shape[:2] == srcMask.shape, "Size must be equal to eachother."
        H, W = srcImage.shape[:2]
        if H < W:
            _H = targetSize
            _W = round(W * (targetSize / H))
        else:
            _W = targetSize
            _H = round(H * (targetSize / W))

        dstImage, dstMask = cv2.resize(srcImage, (_W, _H), interpolation=cv2.INTER_CUBIC), cv2.resize(srcMask, (_W, _H), interpolation=cv2.INTER_NEAREST)

        dstImageFile, dstMaskFile = ChangeFolder(srcImageFile, imageDstFolder), ChangeFolder(srcMaskFile, maskDstFolder)
        cv2.imwrite(dstImageFile, dstImage)
        cv2.imwrite(dstMaskFile , dstMask )

        print(f"Saved [{srcImageFile}] -> [{dstImageFile}] &  [{srcMaskFile}] -> [{dstMaskFile}]")


if __name__ == "__main__":

    # ResizeDataset()
    ExtractAndSaveVisualFeatures(extractor="VQGAN")