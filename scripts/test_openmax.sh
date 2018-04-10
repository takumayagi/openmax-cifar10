#! /bin/sh
#
# test_openmax.sh
# Copyright (C) 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

VDIR=outputs/convnet_val
SDIR=outputs/convnet

python calc_score.py --dataset cifar10 --netType convnet --c10_path data/cifar-10-batches-py --c100_path data/cifar-100-python --save $SDIR --resume $SDIR/model_trial1.npz
python test_openmax.py --save $VDIR --score_path $SDIR/test_scores.joblib --tailsize $1 --alpha $2 --threshold $3
