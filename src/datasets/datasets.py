import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.rle import rle2mask


class CBDataset(Dataset):
    def __init__(self, config, split, transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        self.train_df = pd.read_csv(self.config.TRAIN_DF, engine='python')

        fold_df = pd.read_csv(self.config.FOLD_DF, engine='python')
        self.fold_df = fold_df.loc[fold_df['split'] == self.split].reset_index(drop=True)

        if self.config.DEBUG:
            self.fold_df = self.fold_df[:40]
        print(self.split, 'set:', len(self.fold_df))

        # if config.SAMPLER == 'stratified':
        #     self.labels = self.fold_df['ClassIds'].values
        print('here after initializing Dataset')

    def __len__(self):
        return len(self.fold_df)

    def __getitem__(self, idx):
        print('here in datasets.py getitem')
        ImageId = self.fold_df["ImageId"][idx]
        image = np.load(os.path.join(self.config.PREPROCESSED_DIR, ImageId + '.npz'))['img'] # grayscale
        if self.config.MODEL.IN_CHANNELS == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        print('here after reading Image')

        mask = np.zeros((self.config.DATA.IMG_H, self.config.DATA.IMG_W, 8), dtype=np.uint8)
        EncodedPixels = self.train_df.loc[self.train_df['ImageId_ClassName'].apply(lambda x: x.split('_')[0]) == ImageId]['EncodedPixels'].values
        print('here after reading rle')

        if len(EncodedPixels) > 0:
            for i in range(8):
                if str(EncodedPixels[i]) != 'nan':
                    mask_c = rle2mask(EncodedPixels[i], shape=(self.config.DATA.IMG_H, self.config.DATA.IMG_W))
                    mask[:,:,i] = mask_c
        print('here after making mask')

        # mask의 값은 0과 1!!!!
        # mask = mask * 255 # albu 넣을 때 1로 넣어도 되는지 아직 모름

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

        print('here after normalizing')
        if self.config.MODEL.IN_CHANNELS == 3:
            image = torch.from_numpy(image).permute((2, 0, 1)).float()
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()

        # mask = mask / 255.

        mask = torch.from_numpy(mask).permute((2, 0, 1)).float()
        print('here after making data to tensor')

        return image, mask


class CBDatasetTest(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform

        self.Images = os.listdir(self.config.PREPROCESSED_TEST_DIR)
        self.ImageIds = [f.split('.')[0] for f in self.Images]

        print('Test Images:', len(self.Images))

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, idx):
        imageid = self.ImageIds[idx]
        image = np.load(os.path.join(self.config.PREPROCESSED_TEST_DIR, Image))['img'] # grayscale
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

        return imageid, image
