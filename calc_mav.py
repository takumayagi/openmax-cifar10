#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import argparse
import _pickle as cPickle

import joblib

import numpy as np
import chainer
from chainer import cuda, serializers

import models
from dataset import ImageDataset


def load_dataset(opt):
    def unpickle(fn):
        with open(fn, 'rb') as f:
            data = cPickle.load(f, encoding='latin1')
        return data

    if opt.dataset == 'cifar10':
        train = [unpickle(os.path.join(opt.data, 'data_batch_{}'.format(i))) for i in range(1, 6)]
        train_images = np.concatenate([d['data'] for d in train]).reshape((-1, 3, 32, 32))
        train_labels = np.concatenate([d['labels'] for d in train])
    else:
        train = unpickle(os.path.join(opt.data, 'train'))
        train_images = train['data'].reshape(-1, 3, 32, 32)
        train_labels = train['fine_labels']

    if opt.val:
        rnd = np.random.RandomState(opt.seed)
        val_idxs = rnd.choice(np.arange(len(train_images)), opt.nb_vals, replace=False)
        train_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
        train_images = train_images[train_idxs]
        train_labels = train_labels[train_idxs]

    return train_images, train_labels


def main():
    """
    Compute MAV for all the training examples
    """
    parser = argparse.ArgumentParser(description='BC learning for image classification')
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100'])
    parser.add_argument('--netType', required=True, choices=['convnet'])
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--save', default='None', help='Directory to save the results')
    parser.add_argument('--resume', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--BC', action='store_true', help='BC learning')
    parser.add_argument('--plus', action='store_true', help='Use BC+')
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--nb_vals', type=int, default=10000)
    opt = parser.parse_args()
    opt.nClasses = 10

    model = getattr(models, opt.netType)(opt.nClasses)
    serializers.load_npz(opt.resume, model)
    model.to_gpu(opt.gpu)

    train_images, train_labels = load_dataset(opt)
    train_data = ImageDataset(train_images, train_labels, opt, train=False)
    train_iter = chainer.iterators.SerialIterator(train_data, opt.batchSize, repeat=False, shuffle=False)

    chainer.config.train = False
    chainer.config.enable_backprop = False

    scores = [[] for _ in range(opt.nClasses)]
    for i, batch in enumerate(train_iter):
        x_array, t_array = chainer.dataset.concat_examples(batch)
        x = chainer.Variable(cuda.to_gpu(x_array, opt.gpu))
        fc6 = cuda.to_cpu(model(x).data)  # (B, 10)
        for score, (x, t) in zip(fc6, batch):
            if np.argmax(score) == t:
                scores[t].append(score)

    # Add channel axis (needed at multi-crop evaluation)
    scores = [np.array(x)[:, np.newaxis, :] for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)

    joblib.dump(scores, os.path.join(opt.save, "train_scores.joblib"))
    joblib.dump(mavs, os.path.join(opt.save, "mavs.joblib"))


if __name__ == "__main__":
    main()
