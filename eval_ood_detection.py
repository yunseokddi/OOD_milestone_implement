import argparse
import torch
import numpy as np

from detector.detector import Detector
from model.metric import compute_traditional_odd

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--name', required=True, type=str, help='the name of the model trained')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')

parser.add_argument('--gpu', default='1, 2, 3', type=str, help='gpu index')
parser.add_argument('--adv', help='L_inf OOD', action='store_true')
parser.add_argument('--corrupt', help='corrupted OOD', action='store_true')
parser.add_argument('--adv-corrupt', help='comp. OOD', action='store_true')

parser.add_argument('--in-dist-only', help='only evaluate in-distribution', action='store_true')
parser.add_argument('--out-dist-only', help='only evaluate out-distribution', action='store_true')

parser.add_argument('--method', default='msp', type=str, help='scoring function')
parser.add_argument('--cal-metric', help='calculate metric directly', action='store_true')

parser.add_argument('--epsilon', default=8.0, type=float, help='epsilon')
parser.add_argument('--iters', default=40, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')

parser.add_argument('--severity-level', default=5, type=int, help='severity level')

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size')

parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')

parser.set_defaults(argument=True)

args = parser.parse_args()

if __name__ == "__main__":
    method_args = dict()
    adv_args = dict()
    mode_args = dict()

    adv_args['epsilon'] = args.epsilon
    adv_args['iters'] = args.iters
    adv_args['iter_size'] = args.iter_size
    adv_args['severity_level'] = args.severity_level

    mode_args['in_dist_only'] = args.in_dist_only
    mode_args['out_dist_only'] = args.out_dist_only

    out_datasets = ['LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'SVHN']

    if args.method == "msp":
        detector = Detector(args, out_datasets, method_args, adv_args, mode_args)
        detector.detect()

    elif args.method == "odin":
        method_args['temperature'] = 1000.0

        detector = Detector(args, out_datasets, method_args, adv_args, mode_args)
        detector.detect()


    else:
        assert False, 'Not supported method'

    compute_traditional_odd(args.base_dir, args.in_dataset, out_datasets, args.method, args.name)