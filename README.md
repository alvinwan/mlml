# SSGD
Streaming Stochastic Gradient Descent; save for the shuffling
mechanism, the majority of this scheme is already outlined in a starter
task for Professor Benjamin Recht's Lab at UC Berkeley.

# Installation

To use this repository, clone the repository.

    git clone https://github.com/alvinwan/ssgd.git

Place data in your `data/` folder. Note that these files must be in
binary. Each file encodes a numpy matrix, where the last column of a
matrix is the label. The script will by default load from `data/train`
and `data/test`.

> Note that this project is backwards-compatible, down to Python 2 but
static-typing was introduced to comply with PEP 484, Python 3.5.

# Usage

Run `ssgd.py` at the root of the repository to begin using the utility.

    Usage:
        ssgd.py closed --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
        ssgd.py gd --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
        ssgd.py sgd --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
        ssgd.py ssgd --n=<n> --d=<d> --buffer=<buffer> --train=<train> --test=<test> --nt=<nt> [options]
        ssgd.py (closed|gd|sgd|ssgd) mnist
    
    Options:
        --algo=<algo>       Shuffling algorithm to use [default: external_shuffle]
        --buffer=<num>      Size of memory in megabytes (MB) [default: 5]
        --d=<d>             Number of features
        --damp=<damp>       Amount to multiply learning rate by per epoch [default: 0.99]
        --dtype=<dtype>     The numeric type of each sample [default: float64]
        --epochs=<epochs>   Number of passes over the training data [default: 5]
        --eta0=<eta0>       The initial learning rate [default: 1]
        --iters=<iters>     The number of iterations, used for gd and sgd [default: 1000]
        --n=<n>             Number of training samples
        --k=<k>             Number of classes [default: 10]
        --nt=<nt>           Number of testing samples
        --one-hot=<onehot>  Whether or not to use one hot encoding
        --reg=<reg>         Regularization constant [default: 0.1]
        --train=<train>     Path to training data binary [default: data/train]
        --test=<test>       Path to test data [default: data/test]