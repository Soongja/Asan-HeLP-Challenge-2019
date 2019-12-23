import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold

import utils.config
from utils.rle import mask2rle
from utils.tools import log_time

from global_params import TRAIN_DIR, CLASS_NAMES


@log_time
def preprocess(config):

    os.makedirs(os.path.join(config.SUB_DIR, config.PREPROCESSED_DIR), exist_ok=True)

    ImageIds = os.listdir(TRAIN_DIR)
    ImageIds.sort()

    # --------------------- image preprocessing ---------------------
    for ImageId in ImageIds:
        img = sitk.ReadImage(os.path.join(TRAIN_DIR, ImageId, ImageId + '.dcm'))
        img = sitk.GetArrayFromImage(img).astype(np.uint16).squeeze()

        # 순서 고민
        img = cv2.resize(img, (config.DATA.IMG_W, config.DATA.IMG_H), interpolation=cv2.INTER_CUBIC)

        np.savez_compressed(os.path.join(config.SUB_DIR, config.PREPROCESSED_DIR, ImageId + '.npz'), img=img)
    print('[*] Image preprocessing done!')

    # --------------------- create train_df with rle masks ---------------------
    count = 0
    train_df = pd.DataFrame(columns=['ImageId_ClassName', 'EncodedPixels'])
    for ImageId in ImageIds:
        for ClassName in CLASS_NAMES:
            imageid_classname = ImageId + '_' + ClassName
            mask = cv2.imread(os.path.join(TRAIN_DIR, ImageId, imageid_classname + '.png'), 0)

            mask = cv2.resize(mask, (config.DATA.IMG_W, config.DATA.IMG_H))
            mask = cv2.threshold(mask, 127, 255, 0)[1]

            rle = mask2rle(mask)

            train_df.loc[count] = [imageid_classname, rle]
            count += 1
    train_df.to_csv(os.path.join(config.SUB_DIR, config.TRAIN_DF), index=False)
    print('[*] train_df created!')

    # --------------------- create fold_df ---------------------
    fold_df = pd.DataFrame(columns=['ImageId', 'split'])
    fold_df['ImageId'] = ImageIds

    x = fold_df['ImageId'].values

    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=2019)
    kf.get_n_splits(x)

    for fold, (train_index, val_index) in enumerate(kf.split(x)):
        if fold == config.N_FOLD:
            # print(fold, len(train_index), len(val_index))
            fold_df['split'].iloc[train_index] = 'train'
            fold_df['split'].iloc[val_index] = 'val'

    fold_df.to_csv(os.path.join(config.SUB_DIR, config.FOLD_DF), index=False)
    print('[*] fold_df created!')


@log_time
def preprocess_test(config):
    os.makedirs(os.path.join(config.SUB_DIR, config.PREPROCESSED_TEST_DIR), exist_ok=True)

    Images = os.listdir(config.TEST_DIR)
    ImageIds.sort()

    for Image in Images:
        img = sitk.ReadImage(os.path.join(config.TEST_DIR, Image))
        img = sitk.GetArrayFromImage(img).astype(np.uint16).squeeze()

        # 순서 고민
        img = cv2.resize(img, (config.DATA.IMG_W, config.DATA.IMG_H), interpolation=cv2.INTER_CUBIC)

        np.savez_compressed(os.path.join(config.SUB_DIR, config.PREPROCESSED_TEST_DIR, Image.split('.')[0] + '.npz'), img=img)

    print('[*] Test Image preprocessing done!')


if __name__ == '__main__':
    config = utils.config.load('configs/base.yml')
    preprocess(config)
    preprocess_test(config)
