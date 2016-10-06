# SSGD
Streaming Stochastic Gradient Descent; save for the shuffling
mechanism, the majority of this scheme is already outlined in a starter
task for Professor Benjamin Recht's Lab at UC Berkeley.

# Installation

To use this repository, clone the repository.

    git clone https://github.com/alvinwan/ssgd.git

Place data in your `data/` folder. Note that these files must be in
CSV format. Each csv must contain the number of samples at the top of 
the file. Each line is then formatted with two comma-separated values, 
where the first is a list of values and the second is a single integer.

The script will by default load from `data/train.csv` and
`data/test.csv`.

> Note that this project is backwards-compatible, down to Python 2 but
static-typing was introduced to comply with PEP 484, Python 3.5.

# Usage

Run `ssgd.py` at the root of the repository to begin using the utility.

    Usage:
        ssgd.py

    Options:
        --epochs=<epochs>  Number of passes over the training data
        --eta0=<eta0>      The initial learning rate
        --damp=<damp>      Amount to multiply learning rate by per epoch
        --blocknum=<num>   Number of samples memory can hold, at maximum
        --train=<train>    Path to train file. (.csv) [default: data/train.csv]
        --test=<test>      Path to test file. (.csv) [default: data/test.csv]
