import torch
import torchvision

from torchvision import transforms
from data_loader.svhn_loader import SVHN


class InDataLoader(object):
    def __init__(self, in_dataset, batch_size):
        self.in_dataset = in_dataset
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_dataloader(self):
        if self.in_dataset == "CIFAR-10":
            self.normalizer = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                                   (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
            self.testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True,
                                                        transform=self.transform)
            self.testloaderIn = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=2)

            self.num_classes = 10
            self.num_reject_classes = 5

        elif self.in_dataset == "CIFAR-100":
            self.normalizer = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                                   (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
            self.testset = torchvision.datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True,
                                                         transform=self.transform)
            self.testloaderIn = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=2)

            self.num_classes = 100
            self.num_reject_classes = 10

        elif self.in_dataset == "SVHN":
            self.normalizer = None
            self.testset = SVHN('/datasets/svhn', split='test', transform=transforms.ToTensor(), download=False)
            self.testloaderIn = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=2)
            self.num_classes = 10
            self.num_reject_classes = 5

        return self.testloaderIn, self.num_classes, self.num_reject_classes, self.normalizer
