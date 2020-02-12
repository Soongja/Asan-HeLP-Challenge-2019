import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import SimpleITK as sitk
import pydicom
from scipy.ndimage.filters import gaussian_filter, median_filter, maximum_filter, minimum_filter

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

    def __len__(self):
        return len(self.fold_df)

    def __getitem__(self, idx):
        ImageId = str(self.fold_df["ImageId"][idx])
        npz = np.load(os.path.join(self.config.SUB_DIR, self.config.PREPROCESSED_DIR, f'{ImageId}.npz'))
        # bit = npz['bit']
        image = npz['img']  # grayscale

        # if image.dtype == np.float64:
        #     image += 240
        #     image /= 2.81525
        #     image *= 8
        #     image = np.uint16(image)
        # else:
        #     if bit == 10:
        #         image *= 8
        #     elif bit == 12:
        #         image *= 2
        # assert image.dtype == np.uint16

        if image.dtype == np.float64:
            image += 240
            image *= 4
            image = np.uint16(image)
        assert image.dtype == np.uint16

        #####################################################################

        mask = np.zeros((8, self.config.DATA.IMG_H, self.config.DATA.IMG_W), dtype=np.uint8)
        EncodedPixels = self.train_df.loc[self.train_df['ImageId_ClassName'].apply(lambda x: x.split('_')[0]) == ImageId]['EncodedPixels'].values

        if len(EncodedPixels) > 0:
            for i in range(8):
                if str(EncodedPixels[i]) != 'nan':
                    mask_c = rle2mask(EncodedPixels[i], shape=(self.config.DATA.IMG_H, self.config.DATA.IMG_W))

                    mask[i] = mask_c

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # normalization
        ######################################################################
        image[image > 10000] = 0
        image = image / 10000.
        image = image * 2 - 1
        ######################################################################

        if self.config.MODEL.IN_CHANNELS == 3:
            image = np.stack([image, image, image], axis=-1)
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

        # dicom = pydicom.filereader.dcmread(os.path.join(TEST_DIR, f'{ImageId}.dcm'))
        # bit = dicom.BitsStored
        # del dicom

        image = sitk.ReadImage(os.path.join(TEST_DIR, f'{ImageId}.dcm'))
        image = sitk.GetArrayFromImage(image).squeeze()

        image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H), interpolation=cv2.INTER_CUBIC)

        # if image.dtype == np.float64:
        #     image += 240
        #     image /= 2.81525
        #     image *= 8
        #     image = np.uint16(image)
        # else:
        #     if bit == 10:
        #         image *= 8
        #     elif bit == 12:
        #         image *= 2
        # assert image.dtype == np.uint16

        if image.dtype == np.float64:
            image += 240
            image *= 4
            image = np.uint16(image)
        assert image.dtype == np.uint16

        if self.transform is not None:
            image = self.transform(image)

        # normalization
        ######################################################################
        image[image > 10000] = 0
        image = image / 10000.
        image = image * 2 - 1
        ######################################################################

        if self.config.MODEL.IN_CHANNELS == 3:
            image = np.stack([image, image, image], axis=-1)
            image = torch.from_numpy(image).permute((2, 0, 1)).float()
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()

        return image


def unsharp_mask(img):
    # img dtype: float (0.0~1.0)

    radius = 5
    amount = 2

    img_filtered = gaussian_filter(img, sigma=radius)

    mask = img - img_filtered

    sharpened = img + mask * amount
    sharpened = np.clip(sharpened, 0.0, 1.0)

    return sharpened
