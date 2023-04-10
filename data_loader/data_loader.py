import torchvision.datasets as datasets
import torch

from torchvision import transforms


class CIFAR10DataLoader(object):
    def __init__(self, transform_train, transform_test, kwargs, args):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    def get_dataloader(self):
        return self.train_loader, self.val_loader


class CIFAR100DataLoader(object):
    def __init__(self, transform_train, transform_test, kwargs, args):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=True, download=True,
                              transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    def get_dataloader(self):
        return self.train_loader, self.val_loader


class SVHNDataLoader(object):
    def __init__(self, transform_train, transform_test, kwargs, args):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.SVHN('./datasets/svhn', train=True, download=True,
                          transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            datasets.SVHN('./datasets/svhn', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    def get_dataloader(self):
        return self.train_loader, self.val_loader
