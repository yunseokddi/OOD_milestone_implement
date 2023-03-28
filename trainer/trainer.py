import numpy as np
import torch

from tqdm import tqdm


class Trainer(object):
    def __init__(self, train_loader, model, criterion, optimizer, epoch):
        self.batch_time = AverageMeter()
        self.nat_losses = AverageMeter()
        self.nat_top1 = AverageMeter()

        self.train_loader = train_loader
        self.model = model.train()
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch

        self.tq0 = tqdm(self.train_loader, total=len(self.train_loader))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
