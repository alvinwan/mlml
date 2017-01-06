"""Memory-Limited Machine Learning Command-Line Utility

This utility allows you try several algorithms on popular datasets, to compare
with memory-limited equivalents.

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
    --buffer=<num>      Size of memory in megabytes (MB) [default: 30]
    --d=<d>             Number of features
    --damp=<damp>       Amount to multiply learning rate by per epoch [default: 0.99]
    --dtype=<dtype>     The numeric type of each sample [default: float16]
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
"""

import docopt
import time

from mlml.algorithm import ClosedForm
from mlml.algorithm import GD
from mlml.algorithm import SGD
from mlml.kernels.functions import RBF
from mlml.kernels.generate import RidgeRegressionKernel
from mlml.loss import RidgeRegression
from mlml.ssgd.algorithm import SSGD
from mlml.ssgd.blocks import bytes_per_dtype
from mlml.utils.data import read_dataset


def main() -> None:
    """Load data and launch training, then evaluate accuracy."""
    arguments = preprocess_arguments(docopt.docopt(__doc__, version='MLML 1.0'))

    if arguments['generate']:
        generate(arguments)
    else:
        train(arguments)


def generate(arguments):
    """Generate a Kernel matrix on disk."""
    train = read_dataset(
        data_hook=arguments['--data-hook'],
        dtype=arguments['--dtype'],
        num_classes=arguments['--k'],
        one_hot=arguments['--one-hot'],
        path=arguments['--train'],
        shape=(arguments['--n'], arguments['--d']),
        subset=arguments['--subset'])

    if arguments['--kernel'] == 'RBF':
        rbf = RBF(1)
        RidgeRegressionKernel(
                function=rbf,
                dtype=arguments['--dtype'],
                num_samples=arguments['--num-per-block'],
                data=train,
                dir='data',
                mem_id=('%d-' % arguments['--subset']) + str(time.time())[-5:],
                reg=arguments['--reg'])\
            .generate_rbf(arguments['--simulated'])\
            .generate_Lambda()
    print(' * Finished generation.')


def train(arguments):
    """Train the specified algorithm."""
    test = read_dataset(
        data_hook=arguments['--data-hook'],
        dtype=arguments['--dtype'],
        num_classes=arguments['--k'],
        one_hot=arguments['--one-hot'],
        path=arguments['--test'],
        shape=(arguments['--nt'], arguments['--d']),
        subset=arguments['--subset'])
    if arguments['--loss'] == 'ridge':
        loss = RidgeRegression(arguments['--reg'])
    else:
        raise NotImplementedError
    if arguments['closed']:
        train, model = ClosedForm.from_arguments(
            arguments, test.X, test.labels, loss=loss)
    elif arguments['gd']:
        train, model = GD.from_arguments(
            arguments, test.X, test.labels, loss=loss)
    elif arguments['sgd']:
        train, model = SGD.from_arguments(
            arguments, test.X, test.labels, loss=loss)
    elif arguments['ssgd']:
        train, model = SSGD.from_arguments(
            arguments, test.X, test.labels, loss=loss)
    elif arguments['hsgd']:
        raise NotImplementedError
    else:
        raise UserWarning('Invalid algorithm specified.')
    train_accuracy = model.accuracy(train.X, train.labels)
    test_accuracy = model.accuracy(test.X, test.labels)
    print('Train:', train_accuracy, 'Test:', test_accuracy)


def preprocess_arguments(arguments) -> dict:
    """Preprocessing arguments dictionary by cleaning numeric values.

    Args:
        arguments: The dictionary of command-line arguments
    """

    if arguments['mnist']:
        arguments['--dtype'] = 'uint8'
        arguments['--train'] = 'data/mnist-%s-60000-train' % arguments['--dtype']
        arguments['--test'] = 'data/mnist-%s-10000-test' % arguments['--dtype']
        arguments['--n'] = 60000
        arguments['--nt'] = 10000
        arguments['--k'] = 10
        arguments['--d'] = 784
        arguments['--one-hot'] = 'true'
        arguments['--data-hook'] = lambda X, Y: (X / 255.0, Y)
    if arguments['spam']:
        arguments['--train'] = 'data/spam-%s-2760-train' % arguments['--dtype']
        arguments['--test'] = 'data/spam-%s-690-test' % arguments['--dtype']
        arguments['--n'] = 2760
        arguments['--nt'] = 690
        arguments['--k'] = 1
        arguments['--d'] = 55
    if arguments['cifar-10']:
        arguments['--dtype'] = 'uint8'
        arguments['--train'] = 'data/cifar-10-%s-50000-train' % arguments['--dtype']
        arguments['--test'] = 'data/cifar-10-%s-10000-test' % arguments['--dtype']
        arguments['--n'] = 50000
        arguments['--nt'] = 10000
        arguments['--k'] = 10
        arguments['--d'] = 3072
        arguments['--one-hot'] = 'true'

    arguments['--damp'] = float(arguments['--damp'])
    arguments['--data-hook'] = arguments.get('--data-hook', lambda *args: args)
    arguments['--epochs'] = int(arguments['--epochs'])
    arguments['--eta0'] = float(arguments['--eta0'])
    arguments['--iters'] = int(arguments['--iters'])
    arguments['--logfreq'] = int(arguments['--logfreq'])
    arguments['--momentum'] = float(arguments['--momentum'])
    arguments['--n'] = int(arguments['--n'])
    arguments['--nthreads'] = int(arguments['--nthreads'])
    arguments['--d'] = int(arguments['--d'])
    arguments['--k'] = int(arguments['--k'])
    arguments['--one-hot'] = arguments['--one-hot'].lower() == 'true'
    arguments['--reg'] = float(arguments['--reg'])
    arguments['--step'] = int(arguments['--step'])
    arguments['--subset'] = int(arguments['--subset'])

    if arguments['--memId']:
        arguments['--data-hook'] = lambda *args: args
        arguments['--train'] = 'data/mem-{memid}-Lambda.tmp'.format(
            memid=arguments['--memId'])
        arguments['--test'] = 'data/mem-{memid}-Lambda.tmp'.format(
            memid=arguments['--memId'])
        if arguments['--subset'] > 0:
            arguments['--n'] = arguments['--nt'] = arguments['--subset']
        arguments['--d'] = arguments['--n']
        arguments['--dtype'] = 'float64'  # todo: allow user override
        arguments['--reg'] = 0

    bytes_total = float(arguments['--buffer']) * (10 ** 6)
    bytes_per_sample = (arguments['--d'] + 1) * bytes_per_dtype(arguments['--dtype'])
    arguments['--num-per-block'] = min(
        int(bytes_total // bytes_per_sample),
        arguments['--n'])
    return arguments


if __name__ == '__main__':
    main()
