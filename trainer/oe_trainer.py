import torch
import time
import os
import numpy as np

from tqdm import tqdm
from tensorboard_logger import configure, log_value
from tqdm.contrib import tzip


class OETrainer(object):
    def __init__(self, train_loader_in, train_loader_out, val_loader, model, criterion, ood_criterion, optimizer, args):
        self.train_loader_in = train_loader_in
        self.train_loader_out = train_loader_out
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.ood_criterion = ood_criterion
        self.optimizer = optimizer
        self.args = args
        self.epoch = self.args.epochs

        self.batch_time = AverageMeter()
        self.nat_in_losses = AverageMeter()
        self.nat_out_losses = AverageMeter()
        self.nat_top1 = AverageMeter()

        self.train_loader_out.dataset.offset = np.random.randint((len(train_loader_out.dataset)))

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
        end = time.time()

        self.model.train()

        tq_train = tqdm(self.train_loader_in, position=1, leave=False)
        for i, (in_set, out_set) in enumerate(zip(tq_train, self.train_loader_out)):
            in_len = len(in_set[0])
            out_len = len(out_set[0])
            target = in_set[1]
            target = target.cuda()

            input = torch.cat((in_set[0], out_set[0]), 0).cuda()

            output = self.model(input)

            # in-distribution data cal
            nat_in_output = output[:in_len]
            nat_in_loss = self.criterion(nat_in_output, target)

            # OE data cal
            nat_out_output = output[in_len:]
            nat_out_loss = self.ood_criterion(nat_out_output)

            # measure accuracy and record loss
            nat_prec1 = self.accuracy(nat_in_output.data, target, topk=(1,))[0]
            self.nat_in_losses.update(nat_in_loss.data, in_len)
            self.nat_out_losses.update(nat_out_loss.data, out_len)
            self.nat_top1.update(nat_prec1, in_len)

            loss = nat_in_loss + self.args.beta * nat_out_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            errors = {
                'Epoch': epoch,
                'Batch time': self.batch_time.avg,
                'In loss': self.nat_in_losses.avg.item(),
                'Out loss': self.nat_out_losses.avg.item(),
                'Prec': self.nat_top1.avg.item()
            }

            tq_train.set_postfix(errors)

        # log to TensorBoard
        if self.args.tensorboard:
            log_value('nat_train_acc', self.nat_top1.avg, epoch)

    def _val_epoch(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.eval()

        end = time.time()

        tq_val = tqdm(self.val_loader, position=2, leave=False)

        for input, target in tq_val:
            input = input.cuda()
            target = target.cuda()

            output = self.model(input)
            loss = self.criterion(output, target)

            prec1 = self.accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            errors = {
                'Epoch': epoch,
                'Batch time': self.batch_time.avg,
                'In loss': losses.avg.item(),
                'Prec': top1.avg.item()
            }

            tq_val.set_postfix(errors)

        # log to TensorBoard
        if self.args.tensorboard:
            log_value('val_loss', losses.avg, epoch)
            log_value('val_acc', top1.avg, epoch)

        return top1.avg

    def save_checkpoint(self, state, epoch):
        """Saves checkpoint to disk"""
        directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
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
