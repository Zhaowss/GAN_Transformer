#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pickle import load
import os
from torch.utils.data import DataLoader, TensorDataset
import torch as t
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import random
import torch.nn as nn
from matplotlib import pyplot as plt
import pickle
import  pandas as pd
def seed_all(seed=2020):
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class AdaWeightedLoss(nn.Module):
    def __init__(self, strategy='linear', device='cpu'):
        super(AdaWeightedLoss, self).__init__()
        self.strategy = strategy
        self.device = device

    def forward(self, input, target, global_step, input_label):
        """
        The reconstruction error will be calculated between x and x', where
        x is a vector of x_dim.

        args:
            input: original values, [bsz,seq,x_dim]
            target: reconstructed values
            global_step: training global step
            strategy: how fast the coefficient w2 shrink to 1.0
        return:
        """
        bsz, seq, x_dim = target.size()

        with t.no_grad():
            y= input_label
        error_matrix = (target - input) ** 2
        b = t.sum(error_matrix, dim=-1)/x_dim
        # a = t.tensor([5 for i in range(128)])
        err = (1-y) * b + y * t.max(t.tensor(0).to(self.device), 5-b)
        return t.sum(err) / (bsz * seq)


def normalize(seq):
    return (seq - np.min(seq)) / (np.max(seq) - np.min(seq))


def anomaly_scoring(values, reconstruction_values):
    scores = []
    for v1, v2 in zip(values, reconstruction_values):
        scores.append(np.sqrt(np.sum((v1 - v2) ** 2)))
    return np.array(scores)

all_scores = list()

def metrics_calculate(values, re_values, labels):
    scores = anomaly_scoring(values, re_values) #l2范式


    preds, _ = evaluate(labels, scores, adj=False)
    preds_, _ = evaluate(labels, scores, adj=True)

    f1 = f1_score(y_true=labels, y_pred=preds)
    pre = precision_score(y_true=labels, y_pred=preds)
    re = recall_score(y_true=labels, y_pred=preds)
    auc = roc_auc_score(y_true=labels, y_score=scores)
    # a = accuracy_score(y_true=labels, y_pred=predicted)
    # cm = confusion_matrix(y_true=labels, y_pred=predicted)
    #
    # print('F1 score is [%.5f ], auc score is %.5f.' % (f1, auc1))
    # print('Precision score is [%.5f], recall score is [%.5f].' % (pre, re))
    # print('a score is [%.5f].' % (a), a)
    # print('loss is ', loss)
    # print(cm)
    #
    # with open("./result.txt", "a") as f:
    #     f.write('\n')
    #     f.write('adv_rate:' + str(adv_rate)+'\n')
    #     f.write('F1 score is :'+str(f1))
    #     f.write('\nauc score is :'+str(auc1))
    #     f.write('\nPrecision score is :'+str(pre))
    #     f.write('\nrecall score is :'+str(re))
    #     f.write('\nconfusion_matrix score is :\n' + str(cm))
    #     f.write('\n')

    f1_ = f1_score(y_true=labels, y_pred=preds_)
    pre_ = precision_score(y_true=labels, y_pred=preds_)
    re_ = recall_score(y_true=labels, y_pred=preds_)

    print('F1 score is [%.5f / %.5f] (before adj / after adj), auc score is %.5f.' % (f1, f1_, auc))
    print('Precision score is [%.5f / %.5f], recall score is [%.5f / %.5f].' % (pre, pre_, re, re_))
    all_scores.append(scores[-200:])


def evaluate(labels, scores, step=2000, adj=True):
    # best f1
    min_score = min(scores)
    max_score = max(scores)
    best_f1 = 0.0
    best_preds = None
    for th in tqdm(np.linspace(min_score, max_score, step), ncols=70):
        preds = (scores > th).astype(int)
        if adj:
            preds = adjust_predicts(labels, preds)
        f1 = f1_score(y_true=labels, y_pred=preds)
        if f1 > best_f1:
            best_f1 = f1
            best_preds = preds

    return best_preds, best_f1


def adjust_predicts(label, pred=None):
    predict = pred.astype(bool)
    actual = label > 0.1
    anomaly_state = False
    for i in range(len(label)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
        elif not actual[i]:
            anomaly_state = False

        if anomaly_state:
            predict[i] = True
    return predict.astype(int)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = load(f)
        return data


def get_from_one(ts, window_size, stride):
    ts_length = ts.shape[0]
    samples = []
    for start in np.arange(0, ts_length, stride):
        if start + window_size > ts_length:
            break
        samples.append(ts[start:start+window_size])
    return np.array(samples)


def remove_all_same(train_x, test_x):
    remove_idx = []
    for col in range(train_x.shape[1]):
        if max(train_x[:, col]) == min(train_x[:, col]):
            remove_idx.append(col)
        else:
            train_x[:, col] = normalize(train_x[:, col])

        if max(test_x[:, col]) == min(test_x[:, col]):
            remove_idx.append(col)
        else:
            test_x[:, col] = normalize(test_x[:, col])

    all_idx = set(range(train_x.shape[1]))
    remain_idx = list(all_idx - set(remove_idx))
    return train_x[:, remain_idx], test_x[:, remain_idx]


def load_data(data_prefix, val_size, window_size=100, stride=1, batch_size=4, dataloder=False, noise=0, prob = 0):
    # root path
    root_path = './KDDcup2019'
    Data_train=pd.read_csv(os.path.join(root_path, 'Training.csv'),header=None)
    Data_test = pd.read_csv(os.path.join(root_path, 'Testing.csv'), header=None)

    # load data from .pkl file
    train_x = Data_train.iloc[:,1:42]
    train_y = Data_train.iloc[:,0]
    test_x = Data_test.iloc[:,1:42]
    test_y =Data_test.iloc[:,0]

    # remove columns have 0 variance
    # train_x, test_x = remove_all_same(train_x, test_x)
    # train_test_split
    nc = train_x.shape[1]
    train_len = int(len(train_x) * (1-val_size))
    val_x = train_x[train_len:]
    val_y = train_y[train_len:]
    train_x = train_x[:train_len]
    train_y = train_y[:train_len]

    print('Training data:', train_x.shape)
    print('Validation data:', val_x.shape)
    print('Testing data:', test_x.shape)

    if dataloder:
        # windowed data
        train_x = get_from_one(np.array(train_x),window_size=1,stride=1)
        train_y = get_from_one(np.array(train_y),window_size=1,stride=1)
        print('Training data:', train_x.shape)


        train_dataset = TensorDataset(t.Tensor(train_x), t.LongTensor(train_y))


        data_loader = {"train": DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False),
            "val": (val_x, val_y),
            "test": (test_x, test_y),
            "nc": nc
        }

        return data_loader
    else:
        return {
            "train": train_x,
            "val": (val_x, val_y),
            "test": (test_x, test_y),
            "nc": nc
        }

def get_memory_loss(memory_att, y):
    """The memory attribute should be with size [batch_size, memory_dim, reduced_time_dim, f_h, f_w]
    loss = \sum_{t=1}^{reduced_time_dim} (-mem) * (mem + 1e-12).log()
    averaged on each pixel and each batch
    2. average over batch_size * fh * fw
    """


    s = memory_att.shape
    memory_att = memory_att.permute(0, 2, 1)
    y = y.reshape(s[0], s[-1], 1)
    memory_att = memory_att * (1 - y)
    memory_att = memory_att.permute(0, 2, 1)
    memory_att = (-memory_att) * (memory_att + 1e-12).log()  # [batch_size, memory_dim, time, fh, fw]
    memory_att = memory_att.sum() / (s[0] * s[-1])
    return memory_att
