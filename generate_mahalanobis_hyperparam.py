import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
import numpy as np
import argparse
import model.densenet as dn
import model.wideresnet as wn
import warnings
import time

from data_loader.data_loader import CIFAR10DataLoader, CIFAR100DataLoader, SVHNDataLoader
from torch.autograd import Variable
from utils.mahalanobis_lib import sample_estimator, get_mahalanobis_score
from sklearn.linear_model import LogisticRegressionCV
from model.mahalanobis_metric import metric, print_results

warnings.filterwarnings('ignore')

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--name', required=True, type=str,
                    help='neural network name and training set')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')

parser.add_argument('--gpu', default='1,2', type=str,
                    help='gpu index')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=10, type=int,
                    help='mini-batch size')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')

parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')

parser.set_defaults(argument=True)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


class GenerateMahalanobisHyperParam(object):
    def __init__(self, args, save_dir):
        self.stypes = ['mahalanobis']
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.args = args
        self.kwargs = {'num_workers': 4, 'pin_memory': True}

        if self.args.in_dataset == "CIFAR-10":
            self.normalizer = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                   std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            data_obj = CIFAR10DataLoader(transform_train=transform, transform_test=transform, kwargs=self.kwargs
                                         , args=self.args)
            self.train_loader_in, self.test_loader_in = data_obj.get_dataloader()

            self.num_classes = 10

        elif self.args.in_dataset == "CIFAR-100":
            self.normalizer = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                                   (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            data_obj = CIFAR100DataLoader(transform_train=transform, transform_test=transform, kwargs=self.kwargs
                                          , args=self.args)
            self.train_loader_in, self.test_loader_in = data_obj.get_dataloader()

            self.num_classes = 100

        elif self.args.in_dataset == "SVHN":
            self.normalizer = None

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            data_obj = SVHNDataLoader(transform_train=transform, transform_test=transform, kwargs=self.kwargs
                                      , args=self.args)
            self.train_loader_in, self.test_loader_in = data_obj.get_dataloader()

            self.args.epochs = 20
            self.num_classes = 10

        else:
            assert False, 'Not supported in_dataset : {}'.format(self.args.in_dataset)

        if self.args.model_arch == "densenet":
            self.model = dn.DenseNet3(self.args.layers, self.num_classes, normalizer=self.normalizer).cuda()


        elif self.args.model_arch == "wideresnet":
            self.model = wn.WideResNet(self.args.depth, self.num_classes,
                                       normalizer=self.normalizer).cuda()


        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)

        checkpoint = torch.load(
            "./checkpoints/{model}/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(
                in_dataset=self.args.in_dataset, model=self.args.model_arch,
                name='vanilla',
                epochs=self.args.epochs))

        self.model.load_state_dict(checkpoint['state_dict'])

        # self.model = torch.nn.DataParallel(self.model).to(device)
        self.model.to(device)
        self.model.eval()

        self.sample_mean, self.precision = self.get_mean_cov()

    def get_mean_cov(self):
        temp_x = torch.rand(2, 3, 32, 32)
        temp_x = Variable(temp_x).cuda()
        temp_list = self.model.feature_list(temp_x)[1]
        self.num_output = len(temp_list)
        feature_list = np.empty(self.num_output)
        count = 0

        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1

        print("Get sample mean and covariance")
        sample_mean, precision = sample_estimator(self.model, self.num_classes, feature_list, self.train_loader_in)

        return sample_mean, precision

    def train_logistic_regression(self):
        print("Train logistic regression model")
        m = 500

        train_in = []
        train_in_label = []
        train_out = []

        val_in = []
        val_in_label = []
        val_out = []

        cnt = 0

        for data, target in self.test_loader_in:
            data = data.numpy()
            target = target.numpy()
            for x, y in zip(data, target):
                cnt += 1
                if cnt <= m:
                    train_in.append(x)
                    train_in_label.append(y)
                elif cnt <= 2 * m:
                    val_in.append(x)
                    val_in_label.append(y)

                if cnt == 2 * m:
                    break
            if cnt == 2 * m:
                break

        print('In', len(train_in), len(val_in))

        criterion = nn.CrossEntropyLoss().cuda()
        adv_noise = 0.05

        for i in range(int(m / args.batch_size) + 1):
            if i * args.batch_size >= m:
                break

            data = torch.tensor(train_in[i * args.batch_size:min((i + 1) * args.batch_size, m)])
            target = torch.tensor(train_in_label[i * args.batch_size:min((i + 1) * args.batch_size, m)])
            data = data.cuda()
            target = target.cuda()

            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = self.model(data)

            self.model.zero_grad()
            inputs = Variable(data.data, requires_grad=True).cuda()
            output = self.model(inputs)
            loss = criterion(output, target)
            loss.backward()

            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
            adv_data = torch.clamp(adv_data, 0.0, 1.0)

            train_out.extend(adv_data.cpu().numpy())

        for i in range(int(m / args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            data = torch.tensor(val_in[i * args.batch_size:min((i + 1) * args.batch_size, m)])
            target = torch.tensor(val_in_label[i * args.batch_size:min((i + 1) * args.batch_size, m)])
            data = data.cuda()
            target = target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = self.model(data)

            self.model.zero_grad()
            inputs = Variable(data.data, requires_grad=True).cuda()
            output = self.model(inputs)
            loss = criterion(output, target)
            loss.backward()

            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
            adv_data = torch.clamp(adv_data, 0.0, 1.0)

            val_out.extend(adv_data.cpu().numpy())

        print('Out', len(train_out), len(val_out))

        train_lr_data = []
        train_lr_label = []
        train_lr_data.extend(train_in)
        train_lr_label.extend(np.zeros(m))
        train_lr_data.extend(train_out)
        train_lr_label.extend(np.ones(m))
        train_lr_data = torch.tensor(train_lr_data)
        train_lr_label = torch.tensor(train_lr_label)

        best_fpr = 1.1
        best_magnitude = 0.0

        for magnitude in [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:
            train_lr_Mahalanobis = []
            total = 0
            for data_index in range(int(np.floor(train_lr_data.size(0) / args.batch_size))):
                data = train_lr_data[total: total + args.batch_size].cuda()
                total += args.batch_size
                Mahalanobis_scores = get_mahalanobis_score(data, self.model, self.num_classes, self.sample_mean,
                                                           self.precision, self.num_output,
                                                           magnitude)
                train_lr_Mahalanobis.extend(Mahalanobis_scores)

            train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
            regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)

            print('Logistic Regressor params:', regressor.coef_, regressor.intercept_)

            t0 = time.time()
            f1 = open(os.path.join(self.save_dir, "confidence_mahalanobis_In.txt"), 'w')
            f2 = open(os.path.join(self.save_dir, "confidence_mahalanobis_Out.txt"), 'w')

            ########################################In-distribution###########################################
            print("Processing in-distribution images")

            count = 0
            for i in range(int(m / args.batch_size) + 1):
                if i * args.batch_size >= m:
                    break
                images = torch.tensor(val_in[i * args.batch_size: min((i + 1) * args.batch_size, m)]).cuda()
                # if j<1000: continue
                batch_size = images.shape[0]
                Mahalanobis_scores = get_mahalanobis_score(images, self.model, self.num_classes, self.sample_mean,
                                                           self.precision,
                                                           self.num_output, magnitude)
                confidence_scores = regressor.predict_proba(Mahalanobis_scores)[:, 1]

                for k in range(batch_size):
                    f1.write("{}\n".format(-confidence_scores[k]))

                count += batch_size
                print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time() - t0))
                t0 = time.time()

                ###################################Out-of-Distributions#####################################
            t0 = time.time()
            print("Processing out-of-distribution images")
            count = 0

            for i in range(int(m / args.batch_size) + 1):
                if i * args.batch_size >= m:
                    break
                images = torch.tensor(val_out[i * args.batch_size: min((i + 1) * args.batch_size, m)]).cuda()
                # if j<1000: continue
                batch_size = images.shape[0]

                Mahalanobis_scores = get_mahalanobis_score(images, self.model, self.num_classes, self.sample_mean,
                                                           self.precision,
                                                           self.num_output, magnitude)

                confidence_scores = regressor.predict_proba(Mahalanobis_scores)[:, 1]

                for k in range(batch_size):
                    f2.write("{}\n".format(-confidence_scores[k]))

                count += batch_size
                print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time() - t0))
                t0 = time.time()

            f1.close()
            f2.close()

            results = metric(self.save_dir, self.stypes)
            print_results(results, self.stypes)
            fpr = results['mahalanobis']['FPR']
            if fpr < best_fpr:
                best_fpr = fpr
                best_magnitude = magnitude
                best_regressor = regressor

        print('Best Logistic Regressor params:', best_regressor.coef_, best_regressor.intercept_)
        print('Best magnitude', best_magnitude)

        print('saving results...')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(os.path.join(self.save_dir, 'results'),
                np.array([self.sample_mean, self.precision, best_regressor.coef_, best_regressor.intercept_,
                          best_magnitude]))

        return True


if __name__ == "__main__":
    save_path = os.path.join("./output/mahalanobis_hyperparams/", args.in_dataset, args.name)
    Generate = GenerateMahalanobisHyperParam(args, save_path)

    if Generate.train_logistic_regression():
        print("FINISH")
