import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from albumentations import (
    OneOf, Compose,
    Flip, ShiftScaleRotate, GridDistortion, ElasticTransform,
    RandomGamma, RandomContrast, RandomBrightness, RandomBrightnessContrast,
    Blur, MedianBlur, MotionBlur, GaussianBlur,
    CLAHE, IAASharpen, GaussNoise,
    RandomSizedCrop, CropNonEmptyMaskIfExists,
    RandomSunFlare,
    HueSaturationValue, RGBShift)


def strong_aug(p=1.0):
    return Compose([
        # Flip(p=0.75),  # ok
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        # RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        # OneOf([
        #     Blur(blur_limit=5, p=1.0),
        #     MedianBlur(blur_limit=5, p=1.0),
        #     MotionBlur(p=1.0),
        # ], p=0.2),
        # OneOf([
        #     HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        #     RGBShift(p=1.0)
        # ], p=0.1),
        # GaussNoise(p=0.1),

        # OneOf([
        #     GridDistortion(p=1.0),
        #     ElasticTransform(p=1.0)
        # ], p=0.2),
        # OneOf([
        #     CLAHE(p=1.0),
        #     IAASharpen(p=1.0),
        # ], p=0.2)
    ], p=p, additional_targets={
            'mask0': 'mask',
            'mask1': 'mask',
            'mask2': 'mask',
            'mask3': 'mask',
            'mask4': 'mask',
            'mask5': 'mask',
            'mask6': 'mask',
            'mask7': 'mask'
    })


class Albu():
    def __call__(self, image, mask):
        augmentation = strong_aug()

        data = {"image": image,
                "mask0": mask[0], "mask1": mask[1], "mask2": mask[2], "mask3": mask[3],
                "mask4": mask[4], "mask5": mask[5], "mask6": mask[6], "mask7": mask[7]}
        augmented = augmentation(**data)

        image, mask = augmented["image"], \
                      np.stack([augmented["mask0"], augmented["mask1"], augmented["mask2"], augmented["mask3"],
                                augmented["mask4"], augmented["mask5"], augmented["mask6"], augmented["mask7"]], axis=0)

        return image, mask


class Albu_test():
    def __call__(self, image, mask):
        augmentation = Compose([
                            # Flip(p=0.75),
                            # RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
                            ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
                            # RandomSizedCrop(min_max_height=(1000,1400), height=384, width=576, w2h_ratio=1.5, p=1.0)
                            # CropNonEmptyMaskIfExists(height=300, width=500, p=0.5)
                            # Blur(blur_limit=5, p=0.5),
                            # MedianBlur(blur_limit=5, p=0.5),
                            # MotionBlur(p=0.5),
                            # GaussianBlur(blur_limit=5, p=0.5),
                            # GaussNoise(p=0.5),
                            # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                            # RGBShift(p=1.0)
                            # GridDistortion(p=1.0),
                            # ElasticTransform(p=1.0)
                            # CLAHE(p=0.5),
                            # IAASharpen(p=0.5)
                            # RandomSunFlare(p=0.5)
                        ], p=1.0, additional_targets={
            'mask0': 'mask',
            'mask1': 'mask',
            'mask2': 'mask',
            'mask3': 'mask',
            'mask4': 'mask',
            'mask5': 'mask',
            'mask6': 'mask',
            'mask7': 'mask'
    })

        data = {"image": image,
                "mask0": mask[0], "mask1": mask[1], "mask2": mask[2], "mask3": mask[3],
                "mask4": mask[4], "mask5": mask[5], "mask6": mask[6], "mask7": mask[7]}
        augmented = augmentation(**data)

        image, mask = augmented["image"], \
                      np.stack([augmented["mask0"], augmented["mask1"], augmented["mask2"], augmented["mask3"],
                                augmented["mask4"], augmented["mask5"], augmented["mask6"], augmented["mask7"]], axis=0)

        return image, mask


if __name__ == "__main__":
    import SimpleITK as sitk

    aug = Albu_test()

    img = sitk.ReadImage('JPCLN001.dcm')
    img = sitk.GetArrayFromImage(img).astype(np.uint16).squeeze()

    mask1 = cv2.imread('JPCLN001_heart border.png', 0)
    mask2 = cv2.imread('JPCLN001_heart.png', 0)
    mask3 = cv2.imread('JPCLN001_left clavicle.png', 0)
    mask4 = cv2.imread('JPCLN001_left lung border.png', 0)
    mask5 = cv2.imread('JPCLN001_left lung.png', 0)
    mask6 = cv2.imread('JPCLN001_right clavicle.png', 0)
    mask7 = cv2.imread('JPCLN001_right lung border.png', 0)
    mask8 = cv2.imread('JPCLN001_right lung.png', 0)

    size = (256, 256)
    img = cv2.resize(img, size)
    mask1 = cv2.resize(mask1, size, interpolation=cv2.INTER_NEAREST)
    mask2 = cv2.resize(mask2, size, interpolation=cv2.INTER_NEAREST)
    mask3 = cv2.resize(mask3, size, interpolation=cv2.INTER_NEAREST)
    mask4 = cv2.resize(mask4, size, interpolation=cv2.INTER_NEAREST)
    mask5 = cv2.resize(mask5, size, interpolation=cv2.INTER_NEAREST)
    mask6 = cv2.resize(mask6, size, interpolation=cv2.INTER_NEAREST)
    mask7 = cv2.resize(mask7, size, interpolation=cv2.INTER_NEAREST)
    mask8 = cv2.resize(mask8, size, interpolation=cv2.INTER_NEAREST)

    mask = np.stack([mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8], axis=0)
    print(mask.shape)

    for i in range(100):
        out_image, out_mask = aug(img, mask)
        print(out_image.dtype)
        out_image = (out_image - np.min(out_image)) / (np.max(out_image) - np.min(out_image))

        cv2.imshow('img', out_image)
        cv2.imshow('mask1', out_mask[0])
        cv2.imshow('mask2', out_mask[1])
        cv2.imshow('mask3', out_mask[2])
        cv2.imshow('mask4', out_mask[3])
        cv2.imshow('mask5', out_mask[4])
        cv2.imshow('mask6', out_mask[5])
        cv2.imshow('mask7', out_mask[6])
        cv2.imshow('mask8', out_mask[7])
        cv2.waitKey()
