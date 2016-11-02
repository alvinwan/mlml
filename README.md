# Memory-Limited Machine Learning (MLML)

This Python3 package offers support for a variety of algorithms on
memory-limited infrastructure. Specifically, this package addresses
three memory-limited scenarios:

1. Dataset is too large to fit in memory.
2. Kernel is too large to fit in memory.
3. Both the dataset and the kernel are too large to fit in memory. (WIP)

created by [Alvin Wan](http://alvinwan.com), with guidance of 
Vaishaal Shankar under 
[Professor Benjamin Recht](https://people.eecs.berkeley.edu/~brecht/) 
at UC Berkeley

This package is split into two sub-packages:

1. `mlml.ssgd`: **Streaming Stochastic Gradient Descent** handles 
datasets too large for memory. Only the necessary portions of 
the dataset are loaded into memory, and to optimize time needed for
disk I/O, data is shuffled on disk and then read sequentially.

2. `mlml.kernel`: To handle kernels too large for memory, this package
has two alternatives. The first is a lazy generator, which generates
inner products needed for computation and then discards unneeded data
using an LCU policy. The second is a part-by-part generator for the
kernel, that then streams the Kernel matrix into memory, as needed.

> Note that this project is backwards-compatible, down to Python 2 but
static-typing was introduced to comply with PEP 484, Python 3.5.

# Installation

To use the included Python utilities, install from PyPi (coming soon).

    pip install mlml

To use the command-line utility, clone the repository.

    git clone https://github.com/alvinwan/mlml.git
    
# Proof of Concept

For now, see [performance.ipynb](https://github.com/alvinwan/mlml/blob/master/demos/performance.ipynb).

# Usage

To use the command-line utility, run `mlml.py` at the root of the 
repository.

    Usage:
    ssgd.py closed --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
    ssgd.py gd --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
    ssgd.py sgd --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
    ssgd.py ssgd --n=<n> --d=<d> --buffer=<buffer> --train=<train> --test=<test> --nt=<nt> [options]
    ssgd.py hsgd --n=<n> --d=<d> --buffer=<buffer> --train=<train> --test=<test> --nt=<nt> [options]
    ssgd.py (closed|gd|sgd|ssgd) (mnist|spam) [options]

    Options:
        --algo=<algo>       Shuffling algorithm to use [default: external_shuffle]
        --buffer=<num>      Size of memory in megabytes (MB) [default: 10]
        --d=<d>             Number of features
        --damp=<damp>       Amount to multiply learning rate by per epoch [default: 0.99]
        --dtype=<dtype>     The numeric type of each sample [default: float64]
        --epochs=<epochs>   Number of passes over the training data [default: 5]
        --eta0=<eta0>       The initial learning rate [default: 1e-6]
        --iters=<iters>     The number of iterations, used for gd and sgd [default: 5000]
        --logfreq=<freq>    Number of iterations between log entries. 0 for no log. [default: 1000]
        --momentum=<mom>    Momentum to apply to changes in weight [default: 0.9]
        --n=<n>             Number of training samples
        --k=<k>             Number of classes [default: 10]
        --nt=<nt>           Number of testing samples
        --one-hot=<onehot>  Whether or not to use one hot encoding [default: False]
        --nthreads=<nthr>   Number of threads [default: 1]
        --reg=<reg>         Regularization constant [default: 0.1]
        --step=<step>       Number of iterations between each alpha decay [default: 10000]
        --train=<train>     Path to training data binary [default: data/train]
        --test=<test>       Path to test data [default: data/test]
        --simulated         Mark memory constraints as simulated. Allows full accuracy tests.
        
# References

- [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.
- F. Niu, B. Recht, C. R ́e, S. J. Wright. [Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf), 2011.
