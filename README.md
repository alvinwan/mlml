# Memory-Limited Machine Learning (MLML)

This Python3 package offers support for a variety of algorithms on
memory-limited infrastructure. Specifically, this package addresses
two memory-limited scenarios:

1. Dataset is too large to fit in memory.
2. Kernel is too large to fit in memory.

created by [Alvin Wan](http://alvinwan.com), with guidance of 
[Vaishaal Shankar](http://vaishaal.com) under 
[Professor Benjamin Recht](https://people.eecs.berkeley.edu/~brecht/) 
at UC Berkeley

This package is split into two sub-packages:

1. `mlml.ssgd`: **Streaming Stochastic Gradient Descent** handles 
datasets too large for memory. Only the necessary portions of 
the dataset are loaded into memory, and to optimize time needed for
disk I/O, data is shuffled on disk and then read sequentially.

2. `mlml.kernel`: To handle kernels too large for memory, this package
generates the kernel matrix part-by-part, performs pre-computation
for common algorithms, and then runs **Kernelized Stochastic Gradient
Descent**, streaming pre-computed matrices into memory as needed.

> Note that this project is backwards-compatible, down to Python 2 but
static-typing was introduced to comply with PEP 484, Python 3.5.

# Usage

## Data Format

The import script can be found at [`mlml/utils/imports.py`](https://github.com/alvinwan/mlml/blob/master/mlml/utils/imports.py). 
Here are its usage details.

    Usage:
        imports.py (mnist|spam|cifar-10) [options]

    Options:
        --dtype=<dtype>             Datatype of generated memmap [default: uint8]
        --percentage=<percentage>   Percentage of data for training [default: 0.8]

To extend this script for other datasets, we recommend using `save_inputs_as_data`
or `save_inputs_lablels_as_data`.

## Scenario 1: Streaming Stochastic Gradient Descent

With `mlml.py`, this algorithm can be run on several popular datasets.
We recommend using the `--simulated` flag when testing with subsets of
your data, so that train accuracy is evaluated on the entire train
dataset.

    python mlml.py ssgd (mnist|spam|cifar-10) [options]
    
For example, the following runs streaming sgd on MNIST with simulated 
memory constraints. Note that the `--buffer` size is in MB.

    python mlml.py ssgd mnist --buffer=1 --simulated

## Scenario 2: Kernelized Stochastic Gradient Descent

> MLML currently prepackages only Kernelized Ridge Regression. However,
there are generic utilities such as `MemMatrix` and extensible interfaces
such as `Loss` and `Model` that enable the addition of custom kernelized 
losses.

With `mlml.py`, there are two steps to solving a kernelized problem; see
the derivation [here.](https://github.com/alvinwan/mlml/blob/master/files/ridgeregression.pdf)
First, generate the kernel matrix and pre-computed matrices. Use the
`--subset=<num>` flag to perform computations on a subset of the data.

    python mlml.py generate (mnist|spam|cifar-10) --kernel=<kernel> [options]

Then, run streaming stochastic gradient to compute the inverse of
our kernel matrix or a function of our kernel matrix.

    python mlml.py ssgd (mnist|spam|cifar-10) --memId=<memId> [options] 

For example, the following runs kernelized sgd on a subset of 10000
samples from MNIST, using the radial basis function (RBF). Note that
the first command will output the `<memId>` needed for the second
command.

    python mlml.py generate mnist --subset=10000 --kernel=RBF
    python mlml.py ssgd mnist --memId=<memId> --subset=2000

## Command-Line Utility

To use the command-line utility, run `mlml.py` at the root of the 
repository.

    Usage:
        mlml.py closed --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
        mlml.py gd --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
        mlml.py sgd --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
        mlml.py ssgd --n=<n> --d=<d> --buffer=<buffer> --train=<train> --test=<test> --nt=<nt> [options]
        mlml.py hsgd --n=<n> --d=<d> --buffer=<buffer> --train=<train> --test=<test> --nt=<nt> [options]
        mlml.py (closed|gd|sgd|ssgd) (mnist|spam|cifar-10) [options]
        mlml.py generate (mnist|spam|cifar-10) --kernel=<kernel> [options]

    Options:
        --algo=<algo>       Shuffling algorithm to use [default: external_shuffle]
        --buffer=<num>      Size of memory in megabytes (MB) [default: 10]
        --d=<d>             Number of features
        --damp=<damp>       Amount to multiply learning rate by per epoch [default: 0.99]
        --dtype=<dtype>     The numeric type of each sample [default: float64]
        --epochs=<epochs>   Number of passes over the training data [default: 3]
        --eta0=<eta0>       The initial learning rate [default: 1e-6]
        --iters=<iters>     The number of iterations, used for gd and sgd [default: 5000]
        --k=<k>             Number of classes [default: 10]
        --kernel=<kernel>   Kernel function to use [default: RBF]
        --loss=<loss>       Type of loss to use [default: ridge]
        --logfreq=<freq>    Number of iterations between log entries. 0 for no log. [default: 1000]
        --memId=<memId>     Id of memory-mapped matrices containing Kernel.
        --momentum=<mom>    Momentum to apply to changes in weight [default: 0.9]
        --n=<n>             Number of training samples
        --nt=<nt>           Number of testing samples
        --one-hot=<onehot>  Whether or not to use one hot encoding [default: False]
        --nthreads=<nthr>   Number of threads [default: 1]
        --reg=<reg>         Regularization constant [default: 0.1]
        --step=<step>       Number of iterations between each alpha decay [default: 10000]
        --train=<train>     Path to training data binary [default: data/train]
        --test=<test>       Path to test data [default: data/test]
        --simulated         Mark memory constraints as simulated. Allows full accuracy tests.
        --subset=<num>      Specify subset of data to pick. Ignored if <= 0. [default: 0]

# Installation

To use the included Python utilities, install from PyPi (coming soon).

    pip install mlml

To use the command-line utility, clone the repository.

    git clone https://github.com/alvinwan/mlml.git

# References

- [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.
- F. Niu, B. Recht, C. R ́e, S. J. Wright. [Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf), 2011.
