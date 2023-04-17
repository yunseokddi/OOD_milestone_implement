import torch
import time
import os

from tqdm import tqdm
from tensorboard_logger import configure, log_value


class OETrainer(object):
    def __init__(self, train_loader, val_loader, model, criterion, optimizer, args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.epoch = self.args.epochs
