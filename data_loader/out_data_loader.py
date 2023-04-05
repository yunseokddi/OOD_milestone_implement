import torch
import torchvision

from torchvision import transforms
from data_loader.svhn_loader import SVHN


class OutDataLoader(object):
    def __init__(self, out_dataset, batch_size):
        self.out_dataset = out_dataset
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_dataloader(self):
        if self.out_dataset == "SVHN":
            self.testsetout = SVHN('datasets/ood_datasets/svhn/', split='test',
                                   transform=transforms.ToTensor(), download=False)
            self.testloaderOut = torch.utils.data.DataLoader(self.testsetout, batch_size=self.batch_size,
                                                             shuffle=True, num_workers=2)

        elif self.out_dataset == 'dtd':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                                          transform=transforms.Compose(
                                                              [transforms.Resize(32), transforms.CenterCrop(32),
                                                               transforms.ToTensor()]))
            self.testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=2)
        elif self.out_dataset == 'places365':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365/test_subset",
                                                          transform=transforms.Compose(
                                                              [transforms.Resize(32), transforms.CenterCrop(32),
                                                               transforms.ToTensor()]))
            self.testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=2)
        else:
            testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(self.out_dataset),
                                                                transform=transforms.Compose(
                                                                    [transforms.Resize(32), transforms.CenterCrop(32),
                                                                     transforms.ToTensor()]))
            self.testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=2)

        return self.testloaderOut
