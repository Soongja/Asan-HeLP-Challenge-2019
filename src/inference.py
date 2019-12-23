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

import utils.config
import utils.checkpoint
from models.model_factory import get_model
from datasets.dataloader import get_test_dataloader
from utils.tools import log_time
from global_params import TEST_DIR, VOLUME_DIR, OUTPUT_DIR, CLASS_NAMES


# Global params - mounted diferrently based on phase 1 or 2.
#################################################################

ImageIds = [f.split('.')[0] for f in os.listdir(TEST_DIR)]
ImageIds.sort()

OriginalSizes = []
for ImageId in ImageIds:
    dcm = sitk.ReadImage(os.path.join(TEST_DIR, f'{ImageId}.dcm'))
    size = dcm.GetSize()[:2]
    OriginalSizes.append(size)
    del dcm

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
    print('inference finished. shape:', output.shape)
    return output


def run(config):
    model = get_model(config, pretrained=False).cuda()

    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, score, loss = utils.checkpoint.load_checkpoint(config, model, checkpoint)
    else:
        raise Exception('[*] no checkpoint found')

    test_loader = get_test_dataloader(config, ImageIds, transform=None)
    out = inference(model, test_loader)

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


def postprocess(probability, original_size, threshold=0.5, min_size=0, min_coverage=0, fill_up=False):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    mask = np.uint8(mask)

    predictions = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    if fill_up:
        contours, _ = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_up_predictions = np.zeros((original_size[1], original_size[0]), np.uint8)
        for c in contours:
            cv2.drawContours(filled_up_predictions, [c], 0, 1, -1)
        predictions = filled_up_predictions

    if min_size:
        num_component, component = cv2.connectedComponents(predictions.astype(np.uint8))
        predictions = np.zeros((original_size[1], original_size[0]), np.uint8)
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                predictions[p] = 1

    # if min_coverage:
        # if np.sum(predictions) / predictions.size < min_coverage:
        #     predictions[:,:] = 0

    predictions *= 255

    return predictions


# def change_final_shape(input, out_hw=(350,525)):
#     # input shape: [N,C,H,W]
#     output = np.zeros((input.shape[0], input.shape[1], out_hw[0], out_hw[1]), np.float32)
#
#     for i in range(input.shape[0]):
#         for j in range(input.shape[1]):
#             output[i,j] = cv2.resize(input[i,j], (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)
#
#     return output


@log_time
def main():
    import warnings
    warnings.filterwarnings("ignore")
    seed_everything()

    folds = []
    ymls = ['configs/base.yml']
    for yml in ymls:
        config = utils.config.load(yml)

        fold = run(config)
        folds.append(fold)

    # fold shape: (N, 8, 256, 256)
    for fold in folds:
        for idx in range(fold.shape[0]):
            os.makedirs(os.path.join(OUTPUT_DIR, ImageIds[idx]), exist_ok=True)
            for c in range(8):
                out = postprocess(fold[idx][c], OriginalSizes[idx])
                cv2.imwrite(os.path.join(OUTPUT_DIR, ImageIds[idx], f'{ImageIds[idx]}_{CLASS_NAMES[c]}.png'), out)

    print('success!')


if __name__ == '__main__':
    raise Exception('not implemented yet')
    # main()
