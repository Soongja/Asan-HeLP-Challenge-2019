import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

import utils.config
from utils.rle import rle2mask
from utils.tools import log_time

from global_params import TRAIN_DIR, CLASS_NAMES


@log_time
def eda(config):

    ImageIds = os.listdir(TRAIN_DIR)
    ImageIds.sort()

    print('-----------Start image EDA-----------')
    print('ImageId    min    max   mean    med       shape')
    for ImageId in ImageIds:
        img = sitk.ReadImage(os.path.join(TRAIN_DIR, ImageId, ImageId + '.dcm'))
        img = sitk.GetArrayFromImage(img).astype(np.uint16).squeeze()
        h, w = img.shape

        print(f'{ImageId}  {int(np.min(img)):5d}  {int(np.max(img)):5d}  {int(np.mean(img)):5d}  {int(np.median(img)):5d}'
              f'  ({h},{w})')

    print('-----------Start raw mask EDA-----------')
    m_ratios = []
    for ClassName in CLASS_NAMES:
        m_ratio = 0
        for ImageId in ImageIds:
            imageid_classname = ImageId + '_' + ClassName
            mask = cv2.imread(os.path.join(TRAIN_DIR, ImageId, imageid_classname + '.png'), 0)
            mask = mask / 255.
            m_ratio += (np.sum(mask) / mask.size)
        m_ratios.append(m_ratio / len(ImageIds))

    print('   c0    c1    c2    c3    c4    c5    c6    c7')
    print(f'{m_ratios[0]:.4f} {m_ratios[1]:.4f} {m_ratios[2]:.4f} {m_ratios[3]:.4f} {m_ratios[4]:.4f} {m_ratios[5]:.4f} {m_ratios[6]:.4f} {m_ratios[7]:.4f}')


@log_time
def eda_preprocessed_masks(config):

    ImageIds = os.listdir(TRAIN_DIR)
    ImageIds.sort()

    train_df = pd.read_csv(os.path.join(config.SUB_DIR, config.TRAIN_DF), engine='python')

    print('-----------Start preprocessed mask EDA-----------')
    m_ratios = []
    for ClassName in CLASS_NAMES:
        m_ratio = 0
        rles = train_df.loc[train_df['ImageId_ClassName'].apply(lambda x: x.split('_')[1]) == ClassName]['EncodedPixels'].values
        for rle in rles:
            mask = rle2mask(rle, shape=(config.DATA.IMG_H, config.DATA.IMG_W))
            m_ratio += (np.sum(mask) / mask.size)

        m_ratios.append(m_ratio / len(rles))

    print('   c0    c1    c2    c3    c4    c5    c6    c7')
    print(f'{m_ratios[0]:.4f} {m_ratios[1]:.4f} {m_ratios[2]:.4f} {m_ratios[3]:.4f} {m_ratios[4]:.4f} {m_ratios[5]:.4f} {m_ratios[6]:.4f} {m_ratios[7]:.4f}')


if __name__ == '__main__':
    config = utils.config.load('configs/base.yml')
    eda(config)
    eda_preprocessed_masks(config)
