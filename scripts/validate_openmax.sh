#! /bin/sh
#
# validate_openmax.sh
# Copyright (C) 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

SDIR=outputs/convnet_val

python calc_mav.py --dataset cifar10 --netType convnet --data data/cifar-10-batches-py --save $SDIR --resume $SDIR/model_trial1.npz --val
python calc_dist.py --save $SDIR
python calc_score.py --dataset cifar10 --netType convnet --c10_path data/cifar-10-batches-py --c100_path data/cifar-100-python --save $SDIR --resume $SDIR/model_trial1.npz --val
python validate_openmax.py --save $SDIR --score_path $SDIR/val_score.joblib --val

