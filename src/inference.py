import os
import random
import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import SimpleITK as sitk
from albumentations import Compose, RandomBrightnessContrast
from skimage.morphology import skeletonize

import utils.config
import utils.checkpoint
from models.model_factory import get_model
from datasets.dataloader import get_test_dataloader
from utils.tools import log_time
from utils.rle import mask2rle
from global_params import TEST_DIR, VOLUME_DIR, OUTPUT_DIR, CLASS_NAMES
from eda import eda_test
from preprocess import preprocess_test


# Global params - mounted diferrently based on phase 1 or 2.
#################################################################

ImageIds = [f.split('.')[0] for f in os.listdir(TEST_DIR)]
ImageIds.sort()

ImageIds_first = ImageIds[:202]
ImageIds_second = ImageIds[202:]

OriginalSizes = []
for ImageId in ImageIds:
    dcm = sitk.ReadImage(os.path.join(TEST_DIR, f'{ImageId}.dcm'))
    image = sitk.GetArrayFromImage(dcm).squeeze()
    size = image.shape
    OriginalSizes.append(size)
    del dcm

OriginalSizes_first = OriginalSizes[:202]
OriginalSizes_second = OriginalSizes[202:]

#################################################################


def inference(model, dataloader):
    model.eval()

    output = []
    with torch.no_grad():
        start = time.time()
        for i, images in enumerate(dataloader):
            images = images.cuda()
            logits = model(images)
            logits = F.sigmoid(logits)

            probs = logits.detach().cpu().numpy()

            output.append(probs)

            del images, logits, probs
            torch.cuda.empty_cache()

            end = time.time()
            if i % 10 == 0:
                print('[%2d/%2d] time: %.2f' % (i, len(dataloader), end - start))

    output = np.concatenate(output, axis=0)
    print('inference finished. shape:', output.shape, output.dtype)
    return output


########################################################################################################################

class TTA_Brighter():
    def __call__(self, image):
        augmentation = Compose([
            RandomBrightnessContrast(p=1, brightness_limit=(0.02,0.02), contrast_limit=(0.1,0.1), brightness_by_max=True),
            ], p=1)
        augmented = augmentation(image=image)
        image = augmented["image"]

        return image


class TTA_Darker():
    def __call__(self, image):
        augmentation = Compose([
            RandomBrightnessContrast(p=1, brightness_limit=(-0.02,-0.02), contrast_limit=(-0.1,-0.1), brightness_by_max=True),
            ], p=1)
        augmented = augmentation(image=image)
        image = augmented["image"]

        return image

########################################################################################################################


def run(config, img_ids):
    model = get_model(config, pretrained=False).cuda()

    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, score, loss = utils.checkpoint.load_checkpoint(config, model, checkpoint)
    else:
        raise Exception('[*] no checkpoint found')

    test_loader = get_test_dataloader(config, img_ids, transform=None)
    out = inference(model, test_loader)

    # tta
    # print('TTA brighter')
    # test_loader = get_test_dataloader(config, img_ids, transform=TTA_Brighter())
    # out_brighter = inference(model, test_loader)
    # out += out_brighter
    # del out_brighter
    #
    # print('TTA darker')
    # test_loader = get_test_dataloader(config, img_ids, transform=TTA_Darker())
    # out_darker = inference(model, test_loader)
    # out += out_darker
    # del out_darker

    # out = out / 3.0

    return out


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def skeleton_process_fn(prob, original_size, threshold, dilation):
    prob = cv2.resize(prob, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)
    pred = cv2.threshold(prob, threshold, 1, cv2.THRESH_BINARY)[1]
    pred = np.uint8(pred)
    # pred *= 255

    if dilation:
        kernel = np.ones((3, 3), np.uint8)
        pred = cv2.dilate(pred, kernel, iterations=dilation)

    skel = skeletonize(pred)
    skel = np.asarray(skel, dtype=np.uint8) * 255

    return skel


def postprocess(probability, original_size, threshold=0.5, median_blur=False, fill_up=False, max_component=False):
    probability = cv2.resize(probability, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)
    predictions = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    predictions = np.uint8(predictions)
    predictions *= 255

    if median_blur:
        predictions = cv2.medianBlur(predictions, 7)

    if fill_up:
        contours, _ = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_up_predictions = np.zeros((original_size[0], original_size[1]), np.uint8)
        for c in contours:
            cv2.drawContours(filled_up_predictions, [c], 0, 255, -1)
        predictions = filled_up_predictions

    if max_component:
        max_area = 0
        ci = 0
        num_component, component = cv2.connectedComponents(predictions)
        max_predictions = np.zeros((original_size[0], original_size[1]), np.uint8)

        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > max_area:
                max_area = p.sum()
                ci = c

        largest = (component == ci)
        max_predictions[largest] = 255
        predictions = max_predictions

    return predictions


@log_time
def main():
    import warnings
    warnings.filterwarnings("ignore")
    seed_everything()

    # eda_test()
    # preprocess_test()

    ###############################

    skel_thresh = 0.35
    skel_dilation = 0

    # CLASS_NAMES = ['Aortic Knob', 'Carina', 'DAO', 'LAA', 'Lt Lower CB', 'Pulmonary Conus', 'Rt Lower CB', 'Rt Upper CB']

    # thresholds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    thresholds = [0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35]

    median_blurs = [True, True, True, True, True, True, True, True]
    max_components = [False, True, False, False, False, False, False, False]
    # max_components = [True, True, False, False, False, False, False, False]

    ###############################

    for img_ids, original_sizes in [(ImageIds_first, OriginalSizes_first), (ImageIds_second, OriginalSizes_second)]:
        print('================================= half =====================================')
        # skeleton
        ##########################################################################################################

        if 0:
            skel_prob = np.zeros((len(img_ids), 8, 1024, 1024), np.float32)

            skel_ymls = ['configs/ensemble/skel9_ens.yml', 'configs/ensemble/skel9_ens2.yml']
            # skel_ymls = ['configs/ensemble/skel_ens.yml', 'configs/ensemble/skel_ens2.yml',
            #              'configs/ensemble/skel_ens3.yml', 'configs/ensemble/skel_ens4.yml',
            #              ]

            for skel_yml in skel_ymls:
                skel_config = utils.config.load(skel_yml)

                skel_fold = run(skel_config, img_ids)
                skel_prob += skel_fold
                del skel_fold

            skel_prob = skel_prob / float(len(skel_ymls))

        # segmentation
        ##########################################################################################################

        out_prob = np.zeros((len(img_ids), 8, 1024, 1024), np.float32)
        print('created zero out_prob', out_prob.shape)

        ymls = ['configs/infer.yml']
        # ymls = ['configs/ensemble/ens.yml', 'configs/ensemble/ens2.yml', 'configs/ensemble/ens3.yml',
        #         'configs/ensemble/ens4.yml', 'configs/ensemble/ens5.yml', 'configs/ensemble/ens6.yml',]
        for yml in ymls:
            config = utils.config.load(yml)

            fold = run(config, img_ids)
            out_prob += fold
            del fold

        out_prob = out_prob / float(len(ymls))

        ##########################################################################################################

        for idx in range(len(img_ids)):
            os.makedirs(os.path.join(OUTPUT_DIR, img_ids[idx]), exist_ok=True)
            for c in range(8):

                out = postprocess(out_prob[idx][c], original_sizes[idx], threshold=thresholds[c], median_blur=median_blurs[c], fill_up=False, max_component=max_components[c])
                # print('"' + mask2rle(out) + '",')

                if 0:
                    if c not in [1, 2]:
                        skel = skeleton_process_fn(skel_prob[idx][c], original_sizes[idx], skel_thresh, skel_dilation)
                        out = out | skel

                cv2.imwrite(os.path.join(OUTPUT_DIR, img_ids[idx], f'{img_ids[idx]}_{CLASS_NAMES[c]}.png'), out)

    print('success!')


if __name__ == '__main__':
    # raise Exception('not implemented yet')
    main()
