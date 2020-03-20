
# Gaussian Kernel Density Estimator
 
Optimised implementation of gaussian KDE using multiple threads.

## Data
1) MNIST
2) CIFAR

## Language/Library
1) Python3
2) Numpy
3) multiprocessing
4) tqdm

## Directory Lay out

1) GridSearch.py: Code for performing grid search on values of sigma for both datasets. Usage: "python3 gridSearch.py mnist/cifar"

2) run.py: Code for evaluating test performance of KDE given best sigma obtained from grid search. Usage: "python3 run.py mnist/cifar {sigma value}"

3) model.py: Code pertaining to the Gaussian KDE model class

4) data.py: Code handles data I/O for both datasets. Includes code for data visualisation.

5) config.py: paths of both datasets

