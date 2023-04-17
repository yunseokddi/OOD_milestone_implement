import torch
import numpy as np
import argparse
import os
import torchvision.transforms as transforms
import model.densenet as dn
import model.wideresnet as wn
import torch.backends.cudnn as cudnn
import torch.nn as nn

from tensorboard_logger import configure
from data_loader.data_loader import CIFAR10DataLoader, CIFAR100DataLoader, SVHNDataLoader
from data_loader.tiny_image_data_loader import TinyImages
from model.oe_metric import OELoss

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default='1,2,3', type=str, help='which gpu to use')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')
parser.add_argument('--auxiliary-dataset', default='80m_tiny_images',
                    choices=['80m_tiny_images', 'imagenet'], type=str, help='which auxiliary dataset to use')

parser.add_argument('--beta', default=0.5, type=float, help='beta for out_loss')

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--save-epoch', default=10, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--ood-batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', required=True, type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)
directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

torch.manual_seed(1)
np.random.seed(1)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if args.tensorboard:
        configure("runs/%s" % (args.name))

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.in_dataset == "CIFAR-10":
        normalizer = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        data_obj = CIFAR10DataLoader(transform_train=transform_train, transform_test=transform_test, kwargs=kwargs
                                     , args=args)
        train_loader, val_loader = data_obj.get_dataloader()

        lr_schedule = [50, 75, 90]
        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        normalizer = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        data_obj = CIFAR100DataLoader(transform_train=transform_train, transform_test=transform_test, kwargs=kwargs
                                      , args=args)
        train_loader, val_loader = data_obj.get_dataloader()

        lr_schedule = [50, 75, 90]
        num_classes = 100

    elif args.in_dataset == "SVHN":
        normalizer = None
        data_obj = SVHNDataLoader(transform_train=transform_train, transform_test=transform_test, kwargs=kwargs
                                  , args=args)
        train_loader, val_loader = data_obj.get_dataloader()

        args.epochs = 20
        args.save_epoch = 2
        lr_schedule = [10, 15, 18]
        num_classes = 10

    if args.auxiliary_dataset == '80m_tiny_images':
        ood_loader = torch.utils.data.DataLoader(
            TinyImages(transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
            batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    else:
        assert False, "Check auxiliary_dataset's parameter"

    if args.model_arch == "densenet":
        model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer).cuda()


    elif args.model_arch == "wideresnet":
        model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, dropRate=args.droprate, normalizer=normalizer).cuda()


    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # model = model.cuda()
    model = torch.nn.DataParallel(model).to(device)

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    ood_criterion = OELoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)



if __name__ == "__main__":
    main()
