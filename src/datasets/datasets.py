import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import SimpleITK as sitk

from utils.rle import rle2mask
from global_params import TEST_DIR


class CBDataset(Dataset):
    def __init__(self, config, split, transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        self.train_df = pd.read_csv(os.path.join(self.config.SUB_DIR, self.config.TRAIN_DF), engine='python')

        fold_df = pd.read_csv(os.path.join(self.config.SUB_DIR, self.config.FOLD_DF), engine='python')
        self.fold_df = fold_df.loc[fold_df['split'] == self.split].reset_index(drop=True)

        if self.config.DEBUG:
            self.fold_df = self.fold_df[:40]
        print(self.split, 'set:', len(self.fold_df))

        # if config.SAMPLER == 'stratified':
        #     self.labels = self.fold_df['ClassIds'].values

    def __len__(self):
        return len(self.fold_df)

    def __getitem__(self, idx):
        ImageId = str(self.fold_df["ImageId"][idx])
        image = np.load(os.path.join(self.config.SUB_DIR, self.config.PREPROCESSED_DIR, f'{ImageId}.npz'))['img'] # grayscale
        if self.config.MODEL.IN_CHANNELS == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        mask = np.zeros((8, self.config.DATA.IMG_H, self.config.DATA.IMG_W), dtype=np.uint8)
        EncodedPixels = self.train_df.loc[self.train_df['ImageId_ClassName'].apply(lambda x: x.split('_')[0]) == ImageId]['EncodedPixels'].values

        if len(EncodedPixels) > 0:
            for i in range(8):
                if str(EncodedPixels[i]) != 'nan':
                    mask_c = rle2mask(EncodedPixels[i], shape=(self.config.DATA.IMG_H, self.config.DATA.IMG_W))
                    mask[i] = mask_c

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # normalize = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5]),
        # ])
        # image = normalize(image)

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image * 2 - 1

        if self.config.MODEL.IN_CHANNELS == 3:
            image = torch.from_numpy(image).permute((2, 0, 1)).float()
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()

        mask = torch.from_numpy(mask).float()

        return image, mask


class CBDatasetTest(Dataset):
    def __init__(self, config, ImageIds, transform=None):
        self.config = config
        self.ImageIds = ImageIds
        self.transform = transform

        print('Test Images:', len(self.ImageIds))

    def __len__(self):
        return len(self.ImageIds)

    def __getitem__(self, idx):
        ImageId = self.ImageIds[idx]
        image = sitk.ReadImage(os.path.join(TEST_DIR, f'{ImageId}.dcm'))
        image = sitk.GetArrayFromImage(image).astype(np.uint16).squeeze()

        image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H), interpolation=cv2.INTER_CUBIC)

        if self.config.MODEL.IN_CHANNELS == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            image = self.transform(image)

        # normalize = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5]),
        # ])
        # image = normalize(image)

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image * 2 - 1

        if self.config.MODEL.IN_CHANNELS == 3:
            image = torch.from_numpy(image).permute((2, 0, 1)).float()
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()

        return image
