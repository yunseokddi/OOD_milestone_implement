import os
import torch
import model.densenet as dn
import model.wideresnet as wn
import time
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from model.metric import get_msp_score, get_odin_score, get_Mahanobis_score
from data_loader.in_data_loader import InDataLoader
from data_loader.out_data_loader import OutDataLoader
from torch.autograd import Variable

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector(object):
    def __init__(self, args, out_datasets, method_args, adv_args, mode_args):
        self.args = args
        self.base_dir = args.base_dir
        self.in_dataset = args.in_dataset
        self.out_datasets = out_datasets
        self.batch_size = args.batch_size
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

        # In distribution Data Loader
        in_dataloader = InDataLoader(self.in_dataset, self.batch_size)

        self.test_loader_In, self.num_classes, self.num_reject_classes, self.normalizer = in_dataloader.get_dataloader()

        if self.method != "sofl":
            self.num_reject_classes = 0

        if self.method == "rowl" or self.method == "atom" or self.method == "ntom":
            self.num_reject_classes = 1

        self.method_args['num_classes'] = self.num_classes

        # Model definition
        if self.args.model_arch == "densenet":
            self.model = dn.DenseNet3(self.args.layers, self.num_classes + self.num_reject_classes,
                                      normalizer=self.normalizer)

        elif self.args.model_arch == "wideresnet":
            self.model = wn.WideResNet(self.args.depth, self.num_classes + self.num_reject_classes,
                                       widen_factor=self.args.width, normalizer=self.normalizer)


        else:
            assert False, 'Not supported model arch: {}'.format(self.args.model_arch)

        checkpoint = torch.load(
            "./checkpoints/{model}/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(
                in_dataset=self.in_dataset, model=self.args.model_arch,
                name='vanilla',
                epochs=self.epochs))

        # if args.model_arch == 'densenet_ccu' or args.model_arch == 'wideresnet_ccu':
        #     sewhole_model.load_state_dict(checkpoint['state_dict'])
        # else:
        #     model.load_state_dict(checkpoint['state_dict'])

        self.model.load_state_dict(checkpoint['state_dict'])

        if self.method != "mahalanobis":
            self.model = torch.nn.DataParallel(self.model).to(device)
            self.model.eval()

        else:
            self.model.cuda()
            self.model.eval()

        if self.method == "mahalanobis":
            temp_x = torch.rand(2,3,32,32)
            temp_x = Variable(temp_x).cuda()
            temp_list = self.model.feature_list(temp_x)[1]
            num_output = len(temp_list)
            self.method_args['num_output'] = num_output


    def detect(self):
        if not self.mode_args['out_dist_only']:
            print("Processing in-distribution images")
            self.detect_in_distribution()

            if self.mode_args['in_dist_only']:
                return

        print("Processing out-of-distribution images")
        self.detect_out_distribution()

    def detect_in_distribution(self):
        tq_in_distribution = tqdm(self.test_loader_In)

        t0 = time.time()

        f1 = open(os.path.join(self.in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(self.in_save_dir, "in_labels.txt"), 'w')

        count = 0

        for images, labels in tq_in_distribution:
            images = images.cuda()
            labels = labels.cuda()

            curr_batch_size = images.shape[0]

            inputs = images

            scores = self.get_score(inputs)

            for score in scores:
                f1.write("{}\n".format(score))

            outputs = F.softmax(self.model(inputs)[:, :self.num_classes], dim=1)
            outputs = outputs.detach().cpu().numpy()
            preds = np.argmax(outputs, axis=1)
            confs = np.max(outputs, axis=1)

            for k in range(preds.shape[0]):
                g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            count += curr_batch_size

            errors = {
                'Count': count,
                'Time': time.time() - t0
            }

            tq_in_distribution.set_postfix(errors)
            t0 = time.time()

    def detect_out_distribution(self):
        # OOD Data Loader
        for out_dataset in self.out_datasets:
            out_dataloader = OutDataLoader(out_dataset, self.batch_size)
            test_loader_Out = out_dataloader.get_dataloader()

            out_save_dir = os.path.join(self.in_save_dir, out_dataset)

            if not os.path.exists(out_save_dir):
                os.makedirs(out_save_dir)

            f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

            tq_out_distribution = tqdm(test_loader_Out)

            t0 = time.time()

            count = 0

            for images, labels in tq_out_distribution:
                images = images.cuda()
                labels = labels.cuda()

                curr_batch_size = images.shape[0]

                scores = self.get_score(images)

                for score in scores:
                    f2.write("{}\n".format(score))

                count += curr_batch_size

                errors = {'Dataset' :out_dataset,
                    'Count': count,
                    'Time': time.time() - t0
                }

                tq_out_distribution.set_postfix(errors)
                t0 = time.time()

    def get_score(self, inputs, raw_score=False):
        if self.method == "msp":
            scores = get_msp_score(inputs, self.model)

        elif self.method == "odin":
            scores = get_odin_score(inputs, self.model, self.method_args, self.in_dataset,  self.args.model_arch)

        elif self.method == "mahalanobis":
            scores = get_Mahanobis_score(inputs, self.model, self.method_args)


        else:
            assert False, 'Not supported method'

        return scores
