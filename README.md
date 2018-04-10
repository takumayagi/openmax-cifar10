# openmax-cifar10
Training and test code to evaluate OpenMax (https://arxiv.org/abs/1511.06233).

## Base repositories
We slightly modified bc_learning_image (https://github.com/mil-tokyo/bc_learning_image) for the CIFAR10 code.  
Also, we refactored the code of the authors (https://github.com/abhijitbendale/OSDN).

## Dependencies
* Python 3+
* numpy
* scipy
* joblib
* libmr
* chainer (v2.0.0+)

## Usage
### Download dataset
```
sh scripts/download_dataset.sh
```

### Train CNN
```
# For model selection
sh scripts/train_val.sh

# For final evaluation
sh scripts/train.sh
```

### Validation
```
sh scripts/validate_openmax.sh
```

### Evaluation
Below arguments are determined by a rough parameter search.
```
sh scripts/test_openmax.sh 80 3 0.9
```

## Result
We conducted a simple experiment using CIFAR-10/100 dataset.  
* Training: CIFAR-10 training set
* Test: CIFAR-10 test set + CIFAR-100 test set

| Method | CIFAR-10 top-1 | CIFAR-10 F1 | CIFAR-10 + CIFAR-100 top-1 | CIFAR-10 + CIFAR-100 F1 |
|:---|---:|---:|---:|---:|
| Softmax (closed setting) | *6.23* | *0.9376* | N/A | N/A |
| Softmax + thresholding | **8.85** | **0.851** | 0.372 | 0.695 |
| OpenMax | 18.0 | 0.813 | **0.214** | **0.792** |

### References
Yuji Tokozume, Yoshitaka Ushiku, Tatsuya Harada. Between-class Learning for Image Classification.  
The 31st IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

Meta-Recognition: The Theory and Practice of Recognition Score Analysis  
Walter J. Scheirer, Anderson Rocha, Ross J. Micheals, and Terrance E. Boult  
IEEE T.PAMI, V. 33, Issue 8, August 2011, pages 1689 - 1695  
