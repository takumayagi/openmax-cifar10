#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import argparse
import joblib

import numpy as np
import scipy.spatial.distance as spd
from sklearn.metrics import f1_score, accuracy_score

from utils.openmax import fit_weibull, openmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_type', default='eucos')
    parser.add_argument('--euc_scale', type=float, default=5e-3)
    parser.add_argument('--alpha', type=int, default=5)
    parser.add_argument('--tailsize', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--save', default='None', help='Directory to save the results')
    parser.add_argument('--score_path', default='None')
    opt = parser.parse_args()

    categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    means = joblib.load(os.path.join(opt.save, "mavs.joblib"))
    dists = joblib.load(os.path.join(opt.save, "dists.joblib"))
    scores, labels = joblib.load(opt.score_path)

    weibull_model = fit_weibull(means, dists, categories, opt.tailsize, opt.distance_type)
    pred_y, pred_y_o = [], []
    for score in scores:
        so, ss = openmax(weibull_model, categories, score,
                         opt.euc_scale, opt.alpha, opt.distance_type)
        pred_y.append(np.argmax(ss) if np.max(ss) >= opt.threshold else 10)
        pred_y_o.append(np.argmax(so) if np.max(so) >= opt.threshold else 10)

    print("Open Set evaluation:")
    print("Both:")
    print("Accuracy: softmax={}, openmax={}".format(
        accuracy_score(labels, pred_y), accuracy_score(labels, pred_y_o)))
    print("F1 score: softmax={}, openmax={}".format(
        f1_score(labels, pred_y, average="macro"), f1_score(labels, pred_y_o, average="macro")))

    print("CIFAR-10 only:")
    print("Accuracy: softmax={}, openmax={}".format(
        accuracy_score(labels[:10000], pred_y[:10000]), accuracy_score(labels[:10000], pred_y_o[:10000])))
    print("F1 score: softmax={}, openmax={}".format(
        f1_score(labels[:10000], pred_y[:10000], average="macro"),
        f1_score(labels[:10000], pred_y_o[:10000], average="macro")))

    print("Closed Set evaluation:")
    pred_y = np.argmax(scores, axis=2)[:, 0]
    print("Accuracy: softmax={}".format(accuracy_score(labels[:10000], pred_y[:10000])))
    print("F1 score: softmax={}".format(f1_score(labels[:10000], pred_y[:10000], average="macro")))

if __name__ == "__main__":
    main()
