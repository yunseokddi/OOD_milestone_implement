import torch
import time
import os

from tqdm import tqdm
from tensorboard_logger import configure, log_value


class Trainer(object):
    def __init__(self, train_loader, val_loader, model, criterion, optimizer, args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.epoch = self.args.epochs
        self.batch_time = AverageMeter()
        self.nat_losses = AverageMeter()
        self.nat_top1 = AverageMeter()

    def adjust_learning_rate(self, lr_schedule=[50, 75, 90]):
        """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
        lr = self.args.lr
        if self.epoch >= lr_schedule[0]:
            lr *= 0.1
        if self.epoch >= lr_schedule[1]:
            lr *= 0.1
        if self.epoch >= lr_schedule[2]:
            lr *= 0.1
        # log to TensorBoard
        if self.args.tensorboard:
            log_value('learning_rate', lr, self.epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        for epoch in tqdm(range(self.args.start_epoch, self.epoch), position=0, desc='All epochs'):
            self.adjust_learning_rate()
            self._train_epoch(epoch)
            prec1 = self._val_epoch(epoch)

            if (epoch + 1) % self.args.save_epoch == 0:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                }, epoch + 1)

    def _train_epoch(self, epoch):
        tq_train = tqdm(self.train_loader, total=len(self.train_loader), desc='Train epoch : {}'.format(epoch))

        self.model.train()

        end = time.time()

        for input, target in tq_train:
            input = input.cuda()
            target = target.cuda()

            nat_output = self.model(input)
            nat_loss = self.criterion(nat_output, target)
            nat_prec1 = self.accuracy(nat_output.data, target, topk=(1,))[0]
            self.nat_losses.update(nat_loss.data, input.size(0))
            self.nat_top1.update(nat_prec1, input.size(0))

            loss = nat_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.batch_time.update(time.time() - end)
            end = time.time()

            errors = {
                'Epoch': epoch,
                'Batch time': self.batch_time.avg,
                'Train loss': self.nat_losses.avg.item(),
                'Prec': self.nat_top1.avg.item()
            }

            tq_train.set_postfix(errors)

        if self.args.tensorboard:
            log_value('nat_train_loss', self.nat_losses.avg, self.epoch)
            log_value('nat_train_acc', self.nat_top1.avg, epoch)

    def _val_epoch(self, epoch):
        tq_val = tqdm(self.val_loader, total=len(self.val_loader),desc='Val epoch : {}'.format(epoch))
        batch_time = AverageMeter()
        nat_losses = AverageMeter()
        nat_top1 = AverageMeter()

        self.model.eval()

        end = time.time()

        for input, target in tq_val:
            input = input.cuda()
            target = target.cuda()

            nat_output = self.model(input)
            nat_loss = self.criterion(nat_output, target)
            nat_prec1 = self.accuracy(nat_output.data, target, topk=(1,))[0]
            nat_losses.update(nat_loss.data, input.size(0))
            nat_top1.update(nat_prec1, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            errors = {
                'Epoch': epoch,
                'Batch time': batch_time.avg,
                'Train loss': nat_losses.avg.item(),
                'Prec': nat_top1.avg.item()
            }

            tq_val.set_postfix(errors)

        if self.args.tensorboard:
            log_value('val_loss', nat_losses.avg, self.epoch)
            log_value('val_acc', nat_top1.avg, epoch)

        return nat_top1.avg

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def save_checkpoint(self, state, epoch):
        """Saves checkpoint to disk"""
        directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=self.args.in_dataset, name=self.args.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
        torch.save(state, filename)


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
