import glob
import cv2
import numpy as np
from itertools import product

import albumentations as A

from torch.utils.data import Dataset

from utils import *


class Transforms:
    @staticmethod
    def GetTraining(imageSize=128, imageChannel=3):
        return A.Compose([
            A.RandomCrop(imageSize, imageSize),
            A.HorizontalFlip(p=0.5),
            A.Normalize([0.5] * imageChannel, [0.5] * imageChannel, max_pixel_value=255.),
        ])
    
    @staticmethod
    def GetTesting(imageSize=128, imageChannel=3):
        return A.Compose([
            A.CenterCrop(imageSize, imageSize),
            A.Normalize([0.5] * imageChannel, [0.5] * imageChannel, max_pixel_value=255.),
        ])


class ImageLabelTrainDataset(Dataset):
    def __init__(
            self,
            imageFiles         : list[str],
            labelFiles         : list[str],
            trainingTransform  : A.Compose | None               = None,
            extractorTransform : A.Compose | None               = None,
            fixedFeatures      : dict[str, torch.Tensor] | None = None
    ):
        self.imageFiles = imageFiles
        self.labelFiles = labelFiles

        self.trainingTransform  = trainingTransform or Transforms.GetTraining()
        self.extractorTransform = extractorTransform

        self.fixedFeatures = fixedFeatures

    def __getitem__(self, i):
        image = ReadRGBImage(self.imageFiles[i])
        mask  = cv2.imread(self.labelFiles[i], cv2.IMREAD_GRAYSCALE)

        concat = self.trainingTransform(image=image, mask=mask)
        image, mask = concat["image"], concat["mask"]

        if self.fixedFeatures:
            return torch.from_numpy(image), torch.from_numpy(mask), self.fixedFeatures(GetBasename(self.imageFiles[i], True))
        
        toExtractor = self.extractorTransform(image=image)["image"] if self.extractorTransform else image
        return torch.from_numpy(image), torch.from_numpy(mask), torch.from_numpy(toExtractor)

    def __len__(self):
        return len(self.imageFiles)


class ImageLabelTestDataset(Dataset):
    def __init__(
            self,
            imageFiles         : list[str],
            labelFiles         : list[str],
            testingTransform   : A.Compose | None               = None,
            extractorTransform : A.Compose | None               = None,
            fixedFeatures      : dict[str, torch.Tensor] | None = None

    ):
        self.imageFiles = imageFiles
        self.labelFiles = labelFiles

        self.testingTransform   = testingTransform or Transforms.GetTesting()
        self.extractorTransform = extractorTransform

        self.fixedFeatures = fixedFeatures

        self.pairs = list(product(imageFiles, labelFiles))

    def __getitem__(self, i):
        imageFile, labelFile = self.pairs[i]
        image = ReadRGBImage(imageFile)
        mask  = cv2.imread(labelFile, cv2.IMREAD_GRAYSCALE)

        concat = self.testingTransform(image=image, mask=mask)
        image, mask = concat["image"], concat["mask"]

        if self.fixedFeatures:
            return torch.from_numpy(image), torch.from_numpy(mask), self.fixedFeatures(GetBasename(self.imageFiles[i], True))
        
        toExtractor = self.extractorTransform(image=image)["image"] if self.extractorTransform else image
        return torch.from_numpy(image), torch.from_numpy(mask), torch.from_numpy(toExtractor)

    def __len__(self):
        return len(self.pairs)


def MakeDatasets(
        imageFolder        : str              = "data/image",
        labelFolder        : str              = "data/mask",
        validNames         : list[str]        = ["ADE_train_00000004", "ADE_train_00000191", "ADE_train_00000554", "ADE_train_00000555"],
        trainTransform     : A.Compose | None = None,
        validTransform     : A.Compose | None = None,
        extractorTransform : A.Compose | None = None,
        fixedFeatureFile   : str | None       = None
):
    validImages = [f"{name}.jpg" for name in validNames]
    validLabels = [f"{name}.png" for name in validNames]
    
    if fixedFeatureFile:
        fixedFeatureDict = torch.load(fixedFeatureFile, map_location="cpu")
        fixedFeatureNames, fixedFeatureTensors = fixedFeatureDict["filenames"], fixedFeatureDict["features"]
        
        fixedFeatureValidMask = torch.tensor([(name in validNames) for name in fixedFeatureNames])
        fixedFeatureTrains = fixedFeatureTensors[torch.logical_not(fixedFeatureValidMask)]
        fixedFeatureValids = fixedFeatureTensors[fixedFeatureValidMask]

        fixedFeatureNamesNP     = np.array(fixedFeatureNames)
        fixedFeatureValidMaskNP = fixedFeatureValidMask.numpy()

        fixedFeatureTrainDict = {name: feature for name, feature in zip(fixedFeatureNamesNP[np.logical_not(fixedFeatureValidMaskNP)], fixedFeatureTrains)}
        fixedFeatureValidDict = {name: feature for name, feature in zip(fixedFeatureNamesNP[fixedFeatureValidMaskNP                ], fixedFeatureValids)}
    else:
        fixedFeatureTrainDict = None
        fixedFeatureValidDict = None


    trainset = ImageLabelTrainDataset(
        imageFiles         = [f for f in glob.glob(f"{imageFolder}/*.jpg") if GetBasename(f) not in validImages],
        labelFiles         = [f for f in glob.glob(f"{labelFolder}/*.png") if GetBasename(f) not in validLabels],
        trainingTransform  = trainTransform,
        extractorTransform = extractorTransform,
        fixedFeatures      = fixedFeatureTrainDict
    )
    validset = ImageLabelTestDataset(
        imageFiles         = validImages,
        labelFiles         = validLabels,
        testingTransform   = validTransform,
        extractorTransform = extractorTransform,
        fixedFeatures      = fixedFeatureValidDict
    )
    return trainset, validset


if __name__ == '__main__':

    MakeDatasets(fixedFeatureFile="data/feature/ADE20K-outdoor_Features.pth")