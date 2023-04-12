import torch
import numpy as np
import sklearn.covariance
import tqdm

def sample_estimator(model, num_classes, feature_list, train_loader):
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total= 0,0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)

    list_features = []

    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    train_tq = tqdm.tqdm(train_loader, total=len(train_loader))
    # for data, target in train_loader:
    