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
    Equalize, ISONoise, OpticalDistortion, RandomGridShuffle, RandomResizedCrop, IAAPiecewiseAffine, MultiplicativeNoise,
    ImageCompression, IAASuperpixels, Cutout, CoarseDropout, HorizontalFlip, RandomSizedCrop, MaskDropout, IAAAdditiveGaussianNoise,
    HueSaturationValue, RGBShift)


def strong_aug(p=1.0):
    return Compose([
        # HorizontalFlip(p=0.5),
        ShiftScaleRotate(p=0.9, shift_limit=0.3, scale_limit=0.3, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
        RandomBrightnessContrast(p=0.9, contrast_limit=0.15, brightness_limit=0.15, brightness_by_max=True),
        # CoarseDropout(p=0.5, max_holes=15, max_height=40, max_width=40, min_holes=5, min_height=20, min_width=20),
        # MultiplicativeNoise(p=0.3, multiplier=(0.8, 1.2), per_channel=False, elementwise=True),
        # OpticalDistortion(p=0.5, distort_limit=0.3, shift_limit=0.3, border_mode=cv2.BORDER_CONSTANT),
        # OneOf([
        #     Blur(blur_limit=5, p=1.0),
        #     MedianBlur(blur_limit=5, p=1.0),
        #     MotionBlur(p=1.0),
        # ], p=0.3),
        # GridDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT),
        # GaussNoise(p=0.1),
        # OneOf([
        #     GridDistortion(p=1.0, border_mode=cv2.BORDER_CONSTANT),
        #     ElasticTransform(p=1.0, border_mode=cv2.BORDER_CONSTANT)
        # ], p=0.5),
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


########################################################################################################################

def test_aug(p=1.0):
    return Compose([
        # HorizontalFlip(p=0.5),
        # ShiftScaleRotate(p=0.9, shift_limit=0.3, scale_limit=0.3, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
        RandomBrightnessContrast(p=1.0, brightness_limit=(-0.05,-0.05), contrast_limit=0, brightness_by_max=True),
        # RandomSizedCrop(min_max_height=(400,512), height=512, width=512),
        # RandomGamma(p=1.0, gamma_limit=(90,110)),
        # OpticalDistortion(p=0.5, distort_limit=0.3, shift_limit=0.3, border_mode=cv2.BORDER_CONSTANT),
        # OneOf([
        #     Blur(blur_limit=5, p=1.0),
        #     MedianBlur(blur_limit=5, p=1.0),
        #     MotionBlur(p=1.0),
        # ], p=0.3),
        # GridDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT),
        # GaussNoise(p=0.1),
        # ElasticTransform(p=0.5, border_mode=cv2.BORDER_CONSTANT),
        # MultiplicativeNoise(p=1.0, multiplier=(0.8,1.2), per_channel=False, elementwise=True),
        # CoarseDropout(p=1.0, max_holes=20, max_height=15, max_width=15, min_holes=10, min_height=5, min_width=5),
        # IAAAdditiveGaussianNoise(p=0.5)
        # OneOf([
        #     GridDistortion(p=1.0, border_mode=cv2.BORDER_CONSTANT),
        #     ElasticTransform(p=1.0, border_mode=cv2.BORDER_CONSTANT)
        # ], p=0.5),
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


class Albu_test():
    def __call__(self, image, mask):
        augmentation = test_aug()

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

    img = sitk.ReadImage('sample1.dcm')
    img = sitk.GetArrayFromImage(img).astype(np.uint16).squeeze()
    # img = np.float32(img / 30000.)

    mask1 = cv2.imread('sample1_Aortic Knob.png', 0)
    mask2 = cv2.imread('sample1_Carina.png', 0)
    mask3 = cv2.imread('sample1_DAO.png', 0)
    mask4 = cv2.imread('sample1_LAA.png', 0)
    mask5 = cv2.imread('sample1_Lt Lower CB.png', 0)
    mask6 = cv2.imread('sample1_Pulmonary Conus.png', 0)
    mask7 = cv2.imread('sample1_Rt Lower CB.png', 0)
    mask8 = cv2.imread('sample1_Rt Upper CB.png', 0)

    size = (512, 512)
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

    for i in range(100):
        out_image, out_mask = aug(img, mask)
        print(np.max(out_image))
        out_image = np.float32(out_image / 30000.)

        out_mask_total = np.sum(np.uint16(out_mask),axis=0)
        out_mask_total[out_mask_total>255] = 255
        out_mask_total = np.uint8(out_mask_total)

        cv2.imshow('img', out_image)
        # cv2.imshow('mask1', out_mask[0])
        # cv2.imshow('mask2', out_mask[1])
        # cv2.imshow('mask3', out_mask[2])
        # cv2.imshow('mask4', out_mask[3])
        # cv2.imshow('mask5', out_mask[4])
        # cv2.imshow('mask6', out_mask[5])
        # cv2.imshow('mask7', out_mask[6])
        # cv2.imshow('mask8', out_mask[7])
        cv2.imshow('out_mask_total', out_mask_total)
        cv2.imwrite('sample1_total.png', out_mask_total)
        cv2.waitKey()
