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
from sklearn.metrics import f1_score, accuracy_score

from utils.openmax import fit_weibull, openmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--distance_type', default='eucos')
    parser.add_argument('--eu_weight', type=float, default=5e-3)  # Use the same value in the author's code
    parser.add_argument('--save', default='None', help='Directory to save the results')
    opt = parser.parse_args()

    categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    means = joblib.load(os.path.join(opt.save, "mavs.joblib"))
    dists = joblib.load(os.path.join(opt.save, "dists.joblib"))

    if opt.val:
        scores, labels = joblib.load(os.path.join(opt.save, "val_scores.joblib"))
    else:
        scores, labels = joblib.load(os.path.join(opt.save, "test_scores.joblib"))

    tail_best, alpha_best, th_best = None, None, None
    f1_best = 0.0
    for tailsize in [20, 40, 80]:
        weibull_model = fit_weibull(means, dists, categories, tailsize, opt.distance_type)
        for alpha in [3]:
            for th in [0.0, 0.5, 0.75, 0.8, 0.85, 0.9]:
                print(tailsize, alpha, th)
                pred_y, pred_y_o = [], []
                for score in scores:
                    so, ss = openmax(weibull_model, categories, score,
                                     opt.eu_weight, alpha, opt.distance_type)
                    pred_y.append(np.argmax(ss) if np.max(ss) >= th else 10)
                    pred_y_o.append(np.argmax(so) if np.max(so) >= th else 10)

                print(accuracy_score(labels, pred_y), accuracy_score(labels, pred_y_o))
                openmax_score = f1_score(labels, pred_y_o, average="macro")
                print(f1_score(labels, pred_y, average="macro"), openmax_score)
                if openmax_score > f1_best:
                    tail_best, alpha_best, th_best = tailsize, alpha, th
                    f1_best = openmax_score

    print("Best params:")
    print(tail_best, alpha_best, th_best, f1_best)


if __name__ == "__main__":
    main()
