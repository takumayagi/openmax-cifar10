#! /bin/sh
#
# train.sh
# Copyright (C) 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

python main.py --dataset cifar10 --netType convnet --data data/cifar-10-batches-py --save outputs/convnet --nTrials 1
