import os
import torchvision.transforms as transforms
import torchvision
import torch
import torchvision.datasets as datasets
import model.densenet as dn
import model.wideresnet as wn
import time

from data_loader.svhn_loader import SVHN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector(object):
    def __init__(self, args, out_dataset, method_args, adv_args, mode_args):
        self.args = args
        self.base_dir = args.base_dir
        self.in_dataset = args.in_dataset
        self.out_dataset = out_dataset
        self.bath_size = args.batch_size
        self.method = args.method
        self.method_args = method_args
        self.name = args.name
        self.epochs = args.epochs
        self.adv = args.adv
        self.corrupt = args.corrupt
        self.adv_corrupt = args.adv_corrupt
        self.adv_args = adv_args
        self.mode_args = mode_args

        if self.adv:
            self.in_save_dir = os.path.join(self.base_dir, self.in_dataset, self.method, self.name, 'adv',
                                            str(int(self.adv_args['epsilon'])))

        elif self.adv_corrupt:
            self.in_save_dir = os.path.join(self.base_dir, self.in_dataset, self.method, self.name, 'adv_corrupt',
                                            str(int(self.adv_args['epsilon'])))

        elif self.corrupt:
            self.in_save_dir = os.path.join(self.base_dir, self.in_dataset, self.method, self.name, 'corrupt')

        else:
            self.in_save_dir = os.path.join(self.base_dir, self.in_dataset, self.method, self.name, 'nat')

        if not os.path.exists(self.in_save_dir):
            os.makedirs(self.in_save_dir)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if self.in_dataset == "CIFAR-10":
            self.normalizer = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                                   (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
            self.testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True,
                                                        transform=self.transform)
            self.testloaderIn = torch.utils.data.DataLoader(self.testset, batch_size=self.bath_size, shuffle=True,
                                                            num_workers=2)

            self.num_classes = 10
            self.num_reject_classes = 5

        elif self.in_dataset == "CIFAR-100":
            self.normalizer = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                                   (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
            self.testset = torchvision.datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True,
                                                         transform=self.transform)
            self.testloaderIn = torch.utils.data.DataLoader(self.testset, batch_size=self.bath_size, shuffle=True,
                                                            num_workers=2)

            self.num_classes = 100
            self.num_reject_classes = 10

        elif self.in_dataset == "SVHN":
            self.normalizer = None
            self.testset = SVHN('/datasets/svhn', split='test', transform=transforms.ToTensor(), download=False)
            self.testloaderIn = torch.utils.data.DataLoader(self.testset, batch_size=self.bath_size, shuffle=True,
                                                            num_workers=2)
            self.num_classes = 10
            self.num_reject_classes = 5

        if self.method != "sofl":
            self.num_reject_classes = 0

        if self.method == "rowl" or self.method == "atom" or self.method == "ntom":
            self.num_reject_classes = 1

        self.method_args['num_classes'] = self.num_classes

        if self.args.model_arch == "densenet":
            self.model = dn.DenseNet3(self.args.layers, self.num_classes + self.num_reject_classes, normalizer=self.normalizer)

        elif self.args.model_arch == "wideresnet":
            self.model = wn.WideResNet(self.args.depth, self.num_classes + self.num_reject_classes, widen_factor=self.args.width, normalizer=self.normalizer)


        else:
            assert False, 'Not supported model arch: {}'.format(self.args.model_arch)

        checkpoint = torch.load(
            "./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=self.in_dataset, name=self.name,
                                                                                   epochs=self.epochs))

        # if args.model_arch == 'densenet_ccu' or args.model_arch == 'wideresnet_ccu':
        #     sewhole_model.load_state_dict(checkpoint['state_dict'])
        # else:
        #     model.load_state_dict(checkpoint['state_dict'])

        # self.model.load_state_dict(checkpoint['state_dict'])


        self.model = torch.nn.DataParallel(self.model).to(device)
        self.model.eval()

        if not self.mode_args['out_dist_only']:
            t0 = time.time()

            f1 = open(os.path.join(self.in_save_dir, "in_scores.txt"), 'w')
            g1 = open(os.path.join(self.in_save_dir, "in_labels.txt"), 'w')

            print("Processing in-distribution images")




