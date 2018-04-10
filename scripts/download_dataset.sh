#! /bin/sh
#
# download_dataset.sh
# Copyright (C) 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

CDIR=$PWD
mkdir -p data
cd data/
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xvf cifar-10-python.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar xvf cifar-100-python.tar.gz
cd $CDIR
