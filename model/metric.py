import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable


# OOD scoring method
def get_msp_score(inputs, model):
    with torch.no_grad():
        outputs = model(inputs)

    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

    return scores


def get_odin_score(inputs, model, method_args, in_dataset, model_arch):
    temper = method_args['temperature']

    if model_arch == "densenet":
        if in_dataset == "CIFAR-10":
            method_args['magnitude'] = 0.0016
        elif in_dataset == "CIFAR-100":
            method_args['magnitude'] = 0.0012
        elif in_dataset == "SVHN":
            method_args['magnitude'] = 0.0006
    elif model_arch == 'wideresnet':
        if in_dataset == "CIFAR-10":
            method_args['magnitude'] = 0.0006
        elif in_dataset == "CIFAR-100":
            method_args['magnitude'] = 0.0012
        elif in_dataset == "SVHN":
            method_args['magnitude'] = 0.0002
    else:
        assert False, 'Not supported model arch'

    noise_magnitude_1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = Variable(inputs, requires_grad=True)
    outputs = model(inputs)

    max_index_temp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(max_index_temp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    temp_inputs = torch.add(inputs.data, -noise_magnitude_1, gradient)
    outputs = model(Variable(temp_inputs))
    outputs = outputs / temper

    nn_outputs = outputs.data.cpu()
    nn_outputs = nn_outputs.numpy()
    nn_outputs = nn_outputs - np.max(nn_outputs, axis=1, keepdims=True)
    nn_outputs = np.exp(nn_outputs) / np.sum(np.exp(nn_outputs), axis=1, keepdims=True)
    scores = np.max(nn_outputs, axis=1)

    return scores


# metric for OOD
def cal_metric(known, novel, method):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp / tp[0], [0.]])
    fpr = np.concatenate([[1.], fp / fp[0], [0.]])
    results[mtype] = -np.trapz(1. - fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp / denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp) / denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])

    return results


def get_curve(known, novel, method):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known), np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k + num_n + 1], dtype=int)
    fp = -np.ones([num_k + num_n + 1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k + num_n):
        if k == num_k:
            tp[l + 1:] = tp[l]
            fp[l + 1:] = np.arange(fp[l] - 1, -1, -1)
            break
        elif n == num_n:
            tp[l + 1:] = np.arange(tp[l] - 1, -1, -1)
            fp[l + 1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l + 1] = tp[l]
                fp[l + 1] = fp[l] - 1
            else:
                k += 1
                tp[l + 1] = tp[l] - 1
                fp[l + 1] = fp[l]

    j = num_k + num_n - 1
    for l in range(num_k + num_n - 1):
        if all[j] == all[j - 1]:
            tp[j] = tp[j + 1]
            fp[j] = fp[j + 1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results


def print_results(results, in_dataset, out_dataset, name, method):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: ' + in_dataset)
    print('out_distribution: ' + out_dataset)
    print('Model Name: ' + name)
    print('')

    print(' OOD detection method: ' + method)
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100. * results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100. * results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100. * results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100. * results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100. * results['AUOUT']), end='')
    print('')


def compute_traditional_odd(base_dir, in_dataset, out_datasets, method, name):
    print('Natural OOD')
    print('nat_in vs. nat_out')

    known = np.loadtxt(
        '{base_dir}/{in_dataset}/{method}/{name}/nat/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset,
                                                                           method=method, name=name), delimiter='\n')
    known_sorted = np.sort(known)
    num_k = known.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_sorted[round(0.05 * num_k)]

    all_results = []

    total = 0.0

    for out_dataset in out_datasets:
        novel = np.loadtxt(
            '{base_dir}/{in_dataset}/{method}/{name}/nat/{out_dataset}/out_scores.txt'.format(base_dir=base_dir,
                                                                                              in_dataset=in_dataset,
                                                                                              method=method, name=name,
                                                                                              out_dataset=out_dataset),
            delimiter='\n')

        total += novel.shape[0]

        results = cal_metric(known, novel, method)

        all_results.append(results)

    avg_results = compute_average_results(all_results)

    print_results(avg_results, in_dataset, "All", name, method)

    return
