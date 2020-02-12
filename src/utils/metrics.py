import torch
import numpy as np


def dice_coef(preds, labels):
    preds = preds >= 0.5
    labels = labels >= 0.5

    smooth = 1e-6
    intersection = (preds.float() * labels.float()).sum(dim=(2, 3))
    union = preds.float().sum(dim=(2, 3)) + labels.float().sum(dim=(2, 3))
    # class 별로 찍게 하자 [N, C]
    dice = ((2. * intersection + smooth) / (union + smooth)).mean(dim=0)

    #####################################
    tp = (preds) & (labels)
    fp = (preds) & (~labels)
    fn = (~preds) & (labels)
    tn = (~preds) & (~labels)

    tfpn = np.zeros((8, 4), np.int32)

    for i in range(8):
        tfpn[i,0] = tp[:,i].sum().item()
        tfpn[i,1] = fp[:,i].sum().item()
        tfpn[i,2] = fn[:,i].sum().item()
        tfpn[i,3] = tn[:,i].sum().item()

    return dice, tfpn


def dice_coef_numpy(preds, labels):
    preds = preds >= 0.5
    labels = labels >= 0.5

    smooth = 1e-6
    intersection = (np.float32(preds) * np.float32(labels)).sum(axis=(2, 3))
    union = np.float32(preds).sum(axis=(2, 3)) + np.float32(labels).sum(axis=(2, 3))
    # class 별로 찍게 하자 [N, C]
    dice = ((2. * intersection + smooth) / (union + smooth)).mean(axis=0)

    return dice


if __name__ == '__main__':
    import time

    preds = torch.rand(16, 8, 128, 128)
    labels = torch.randint(0, 2, (16, 8, 128, 128))
    # preds = np.random.rand(8, 4, 128, 128)
    # labels = np.random.randint(2, size=(8, 4, 128, 128))

    print(dice_coef(preds, labels))
    # preds = preds.numpy()
    # labels = labels.numpy()
    # print(dice_coef_numpy(preds, labels))

    # print(preds)
    # preds = np.where(preds > 0.5, 1, 0).astype(np.uint8)
    # print(preds)
    # print(preds.dtype)

    # start = time.time()
    # for i in range(1000):
    #     dice = dice_coef(preds, labels)
    # print(time.time() - start)
