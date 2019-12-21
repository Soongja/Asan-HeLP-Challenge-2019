import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

import utils.config
from utils.rle import rle2mask


# ClassNames = ['Aortic Knob', 'Carina', 'DAO', 'LAA', 'Lt Lower CB', 'Pulmonary Conus', 'Rt Lower CB', 'Rt Upper CB']
def eda(config):

    ImageIds = os.listdir(config.TRAIN_DIR)
    ClassNames = [f.split('.')[0].split('_')[-1] for f in os.listdir(os.path.join(config.TRAIN_DIR, ImageIds[0])) if f.endswith('.png')]
    ImageIds.sort()
    ClassNames.sort()
    print(ClassNames)

    print('-----------Start image EDA-----------')
    print('ImageId    min    max   mean    med       shape')
    for ImageId in ImageIds:
        img = sitk.ReadImage(os.path.join(config.TRAIN_DIR, ImageId, ImageId + '.dcm'))
        img = sitk.GetArrayFromImage(img).astype(np.uint16).squeeze()
        h, w = img.shape

        print(f'{ImageId}  {int(np.min(img)):5d}  {int(np.max(img)):5d}  {int(np.mean(img)):5d}  {int(np.median(img)):5d}'
              f'  ({h},{w})')

    print('-----------Start raw mask EDA-----------')
    m_ratios = []
    for ClassName in ClassNames:
        m_ratio = 0
        for ImageId in ImageIds:
            imageid_classname = ImageId + '_' + ClassName
            mask = cv2.imread(os.path.join(config.TRAIN_DIR, ImageId, imageid_classname + '.png'), 0)
            mask = mask / 255.
            m_ratio += (np.sum(mask) / mask.size)
        m_ratios.append(m_ratio / len(ImageIds))

    print('   c0    c1    c2    c3    c4    c5    c6    c7')
    print(f'{m_ratios[0]:.3f} {m_ratios[1]:.3f} {m_ratios[2]:.3f} {m_ratios[3]:.3f} {m_ratios[4]:.3f} {m_ratios[5]:.3f} {m_ratios[6]:.3f} {m_ratios[7]:.3f}')


def eda_preprocessed_masks(config):

    ImageIds = os.listdir(config.TRAIN_DIR)
    ClassNames = [f.split('.')[0].split('_')[-1] for f in os.listdir(os.path.join(config.TRAIN_DIR, ImageIds[0])) if f.endswith('.png')]
    ImageIds.sort()
    ClassNames.sort()
    print(ClassNames)

    train_df = pd.read_csv(config.TRAIN_DF, engine='python')

    print('-----------Start preprocessed mask EDA-----------')
    m_ratios = []
    for ClassName in ClassNames:
        m_ratio = 0
        rles = train_df.loc[train_df['ImageId_ClassName'].apply(lambda x: x.split('_')[1]) == ClassName]['EncodedPixels'].values
        for rle in rles:
            mask = rle2mask(rle, shape=(config.DATA.IMG_H, config.DATA.IMG_W))
            m_ratio += (np.sum(mask) / mask.size)

        m_ratios.append(m_ratio / len(rles))

    print('   c0    c1    c2    c3    c4    c5    c6    c7')
    print(f'{m_ratios[0]:.3f} {m_ratios[1]:.3f} {m_ratios[2]:.3f} {m_ratios[3]:.3f} {m_ratios[4]:.3f} {m_ratios[5]:.3f} {m_ratios[6]:.3f} {m_ratios[7]:.3f}')


if __name__ == '__main__':
    config = utils.config.load('configs/base.yml')
    eda(config)
    eda_preprocssed_masks(config)
