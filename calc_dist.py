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


def compute_channel_distances(mavs, features, eu_weight):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def main():
    """
    Compute category spesific distance distribution
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--eu_weight', type=float, default=5e-3)  # Use the same value in the author's code
    parser.add_argument('--save', default='None', help='Directory to save the results')
    opt = parser.parse_args()

    scores = joblib.load(os.path.join(opt.save, "train_scores.joblib"))
    mavs = joblib.load(os.path.join(opt.save, "mavs.joblib"))
    dists = [compute_channel_distances(mcv, score, opt.eu_weight) for mcv, score in zip(mavs, scores)]

    joblib.dump(dists, os.path.join(opt.save, "dists.joblib"))


if __name__ == "__main__":
    main()
