import torch

import os
import glob
from tqdm import tqdm

from utils import *
from model import VisualExtractor


def ExtractAndSaveVisualFeatures(
        imageFolder  : str = "data/image",
        saveFolder   : str = "data/feature",
        saveFilename : str = "ADE20K-outdoor_Features.pth"
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    extractor = VisualExtractor().to(device)
    extractor.requires_grad_(False)
    extractor.eval()

    preprocess = extractor.GetPreprocess()

    os.makedirs(saveFolder, exist_ok=True)

    with torch.inference_mode():
        filenames, features = [], None
        for imageFile in tqdm(glob.glob(f"{imageFolder}/*")):
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


if __name__ == "__main__":

    ExtractAndSaveVisualFeatures()