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

    if opt.val:
        # CIFAR-10/100 validation set (randomly selected from training set)
        train_c10 = [unpickle(os.path.join(opt.c10_path,
                                       'data_batch_{}'.format(i))) for i in range(1, 6)]
        train_images_c10 = np.concatenate([d['data'] for d in train_c10]).reshape((-1, 3, 32, 32))
        train_labels_c10 = np.concatenate([d['labels'] for d in train_c10])
        rnd = np.random.RandomState(opt.seed)
        val_idxs = rnd.choice(np.arange(len(train_images_c10)), opt.nb_vals, replace=False)
        val_images_c10 = train_images_c10[val_idxs]
        val_labels_c10 = train_labels_c10[val_idxs]

        train_c100 = unpickle(os.path.join(opt.c100_path, 'train'))
        train_images_c100 = train_c100['data'].reshape(-1, 3, 32, 32)
        train_labels_c100 = np.array(train_c100['fine_labels'])
        rnd = np.random.RandomState(opt.seed)
        val_idxs = rnd.choice(np.arange(len(train_images_c100)), opt.nb_vals, replace=False)
        val_images_c100 = train_images_c100[val_idxs]
        val_labels_c100 = train_labels_c100[val_idxs]
    else:
        # CIFAR-10/100 test set
        val_c10 = unpickle(os.path.join(opt.c10_path, 'test_batch'))
        val_images_c10 = val_c10['data'].reshape((-1, 3, 32, 32))
        val_labels_c10 = val_c10['labels']
        val_c100 = unpickle(os.path.join(opt.c100_path, 'test'))
        val_images_c100 = val_c100['data'].reshape((-1, 3, 32, 32))
        val_labels_c100 = val_c100['fine_labels']

    val_images = np.concatenate((val_images_c10, val_images_c100), axis=0)
    val_labels = np.concatenate((val_labels_c10, np.ones(len(val_labels_c100),
                                 dtype=np.int) * opt.nClasses), axis=0)

    return val_images, val_labels


def main():
    parser = argparse.ArgumentParser(description='BC learning for image classification')
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100'])
    parser.add_argument('--netType', required=True, choices=['convnet'])
    parser.add_argument('--c10_path', required=True)
    parser.add_argument('--c100_path', required=True)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--BC', action='store_true', help='BC learning')
    parser.add_argument('--plus', action='store_true', help='Use BC+')
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--nb_vals', type=int, default=10000)

    parser.add_argument('--distance_type', default='eucos')
    parser.add_argument('--eu_weight', type=float, default=5e-3)
    parser.add_argument('--alpha', type=int, default=5)
    parser.add_argument('--tailsize', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--save', default='None', help='Directory to save the results')
    opt = parser.parse_args()
    opt.nClasses = 10

    model = getattr(models, opt.netType)(opt.nClasses)
    serializers.load_npz(opt.resume, model)
    model.to_gpu(opt.gpu)

    val_images, val_labels = load_dataset(opt)
    val_data = ImageDataset(val_images, val_labels, opt, train=False)
    val_iter = chainer.iterators.SerialIterator(val_data, opt.batchSize, repeat=False, shuffle=False)

    chainer.config.train = False
    chainer.config.enable_backprop = False

    scores, labels = [], []
    for i, batch in enumerate(val_iter):
        x_array, t_array = chainer.dataset.concat_examples(batch)
        x = chainer.Variable(cuda.to_gpu(x_array, opt.gpu))
        fc6 = cuda.to_cpu(model(x).data)  # (B, 10)
        for score, (x, t) in zip(fc6, batch):
            scores.append(score)
            labels.append(t)

    # Add channel axis (needed at multi-crop evaluation)
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)
    if opt.val:
        joblib.dump((scores, labels), os.path.join(opt.save, "val_scores.joblib"))
    else:
        joblib.dump((scores, labels), os.path.join(opt.save, "test_scores.joblib"))


if __name__ == "__main__":
    main()
