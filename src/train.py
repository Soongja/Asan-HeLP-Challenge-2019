import os
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from models.model_factory import get_model
from factory.losses import get_loss
from factory.schedulers import get_scheduler
from factory.optimizers import get_optimizer
from factory.transforms import Albu
from datasets.dataloader import get_dataloader

import utils.config
import utils.checkpoint
from utils.metrics import dice_coef
from utils.tools import prepare_train_directories, AverageMeter, log_time
from utils.experiments import LabelSmoother
from preprocess import preprocess
from eda import eda, eda_preprocessed_masks

import yaml
from easydict import EasyDict as edict


def evaluate_single_epoch(config, model, dataloader, criterion, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    score_1 = AverageMeter()
    score_2 = AverageMeter()
    score_3 = AverageMeter()
    score_4 = AverageMeter()
    score_5 = AverageMeter()
    score_6 = AverageMeter()
    score_7 = AverageMeter()
    score_8 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(dataloader):
            images = images.cuda()
            labels = labels.cuda()

            logits = model(images)

            loss = criterion(logits, labels)
            losses.update(loss.item(), images.shape[0])

            preds = F.sigmoid(logits)

            score = dice_coef(preds, labels)
            score_1.update(score[0].item(), images.shape[0])
            score_2.update(score[1].item(), images.shape[0])
            score_3.update(score[2].item(), images.shape[0])
            score_4.update(score[3].item(), images.shape[0])
            score_5.update(score[4].item(), images.shape[0])
            score_6.update(score[5].item(), images.shape[0])
            score_7.update(score[6].item(), images.shape[0])
            score_8.update(score[7].item(), images.shape[0])
            scores.update(score.mean().item(), images.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_EVERY == 0:
                print(f'[{i:2d}/{len(dataloader):2d}] time: {batch_time.sum:.2f}, loss: {loss.item():.6f},'
                      f'score: {score.mean().item():.4f}'
                      f'[{score[0].item():.4f}, {score[1].item():.4f}, {score[2].item():.4f}, {score[3].item():.4f},'
                      f'{score[4].item():.4f}, {score[5].item():.4f}, {score[6].item():.4f}, {score[7].item():.4f}]')

            del images, labels, logits, preds
            torch.cuda.empty_cache()

    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/score', scores.avg, epoch)
    writer.add_scalar('val/score_1', score_1.avg, epoch)
    writer.add_scalar('val/score_2', score_2.avg, epoch)
    writer.add_scalar('val/score_3', score_3.avg, epoch)
    writer.add_scalar('val/score_4', score_4.avg, epoch)
    writer.add_scalar('val/score_5', score_5.avg, epoch)
    writer.add_scalar('val/score_6', score_6.avg, epoch)
    writer.add_scalar('val/score_7', score_7.avg, epoch)
    writer.add_scalar('val/score_8', score_8.avg, epoch)
    print('average loss over VAL epoch: %f' % losses.avg)

    return scores.avg, losses.avg


def train_single_epoch(config, model, dataloader, criterion, optimizer, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    score_1 = AverageMeter()
    score_2 = AverageMeter()
    score_3 = AverageMeter()
    score_4 = AverageMeter()
    score_5 = AverageMeter()
    score_6 = AverageMeter()
    score_7 = AverageMeter()
    score_8 = AverageMeter()

    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        images = images.cuda()
        labels = labels.cuda()

        logits = model(images)

        if config.LABEL_SMOOTHING:
            smoother = LabelSmoother(config.LABEL_SMOOTHING)
            loss = criterion(logits, smoother(labels))
        else:
            loss = criterion(logits, labels)

        losses.update(loss.item(), images.shape[0])

        loss.backward()
        optimizer.step()

        preds = F.sigmoid(logits)

        score = dice_coef(preds, labels)
        score_1.update(score[0].item(), images.shape[0])
        score_2.update(score[1].item(), images.shape[0])
        score_3.update(score[2].item(), images.shape[0])
        score_4.update(score[3].item(), images.shape[0])
        score_5.update(score[4].item(), images.shape[0])
        score_6.update(score[5].item(), images.shape[0])
        score_7.update(score[6].item(), images.shape[0])
        score_8.update(score[7].item(), images.shape[0])
        scores.update(score.mean().item(), images.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_EVERY == 0:
            print(f'[{epoch}/{config.TRAIN.NUM_EPOCHS}][{i:2d}/{len(dataloader):2d}] time: {batch_time.sum:.2f}, loss: {loss.item():.6f},'
                  f'score: {score.mean().item():.4f}'
                  f'[{score[0].item():.4f}, {score[1].item():.4f}, {score[2].item():.4f}, {score[3].item():.4f},'
                  f'{score[4].item():.4f}, {score[5].item():.4f}, {score[6].item():.4f}, {score[7].item():.4f}]'
                  f'lr: {optimizer.param_groups[0]["lr"]:.6f}')

        del images, labels, logits, preds
        torch.cuda.empty_cache()

    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/score', scores.avg, epoch)
    writer.add_scalar('train/score_1', score_1.avg, epoch)
    writer.add_scalar('train/score_2', score_2.avg, epoch)
    writer.add_scalar('train/score_3', score_3.avg, epoch)
    writer.add_scalar('train/score_4', score_4.avg, epoch)
    writer.add_scalar('train/score_5', score_5.avg, epoch)
    writer.add_scalar('train/score_6', score_6.avg, epoch)
    writer.add_scalar('train/score_7', score_7.avg, epoch)
    writer.add_scalar('train/score_8', score_8.avg, epoch)
    print('average loss over TRAIN epoch: %f' % losses.avg)


def train(config, model, train_loader, test_loader, optimizer, scheduler, writer, start_epoch, best_score, best_loss):
    num_epochs = config.TRAIN.NUM_EPOCHS
    model = model.cuda()

    for epoch in range(start_epoch, num_epochs):

        if epoch < config.LOSS.FINETUNE_EPOCH:
            criterion = get_loss(config.LOSS.NAME)
        else:
            criterion = get_loss(config.LOSS.FINETUNE_LOSS)

        train_single_epoch(config, model, train_loader, criterion, optimizer, writer, epoch)

        test_score, test_loss = evaluate_single_epoch(config, model, test_loader, criterion, writer, epoch)

        print('Total Test Score: %.4f, Test Loss: %.4f' % (test_score, test_loss))

        if test_score > best_score:
            best_score = test_score
            print('Test score Improved! Save checkpoint')
            utils.checkpoint.save_checkpoint(config, model, epoch, test_score, test_loss)

        # utils.checkpoint.save_checkpoint(config, model, epoch, test_score, test_loss)

        if config.SCHEDULER.NAME == 'reduce_lr_on_plateau':
            scheduler.step(test_score)
        else:
            scheduler.step()


@log_time
def run(config):
    model = get_model(config, pretrained=True).cuda()

    optimizer = get_optimizer(config, model.parameters())

    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, score, loss = utils.checkpoint.load_checkpoint(config, model, checkpoint)
    else:
        print('[*] no checkpoint found')
        last_epoch, score, loss = -1, -1, float('inf')
    print('last epoch:{} score:{:.4f} loss:{:.4f}'.format(last_epoch, score, loss))

    optimizer.param_groups[0]['initial_lr'] = config.OPTIMIZER.LR

    scheduler = get_scheduler(config, optimizer, last_epoch)

    if config.SCHEDULER.NAME == 'multi_step':
        milestones = scheduler.state_dict()['milestones']
        step_count = len([i for i in milestones if i < last_epoch])
        optimizer.param_groups[0]['lr'] *= scheduler.state_dict()['gamma'] ** step_count

    writer = SummaryWriter(os.path.join(config.VOLUME_DIR, 'logs', config.LOG_DIR))

    # train_loader = get_dataloader(config, 'train', transform=Albu())
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'val')

    train(config, model, train_loader, val_loader, optimizer, scheduler, writer, last_epoch+1, score, loss)


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('start training.')
    seed_everything()

    ymls = ['configs/base.yml']
    for yml in ymls:
        config = utils.config.load(yml)

        if config.PREPROCESS:
            preprocess(config)
        if config.EDA:
            eda(config)
            eda_preprocessed_masks(config)

        prepare_train_directories(config)
        run(config)

    print('success!')


if __name__ == '__main__':
    main()
