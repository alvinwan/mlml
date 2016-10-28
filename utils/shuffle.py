"""Shuffling algorithm implementations for SSGD.

All of the following shuffling algorithms take advantage of spatial locality
by loading data from and writing data to disk sequentially, wherever possible.
"""

import numpy as np

from .blocks import BlockBuffer
from .blocks import BlockScope
from .blocks import BlockWriter

ALGORITHMS = ('external_sort', 'external_shuffle')


def shuffle_train(
        algorithm: str,
        dtype: str,
        n: int,
        num_features: int,
        num_per_block: int,
        train_path: str) -> str:
    """Invoke the correct shuffling algorithm.

    Args:
        algorithm: The shuffling algorithm to use
        dtype: Data type of samples in file
        n: Number of total samples
        num_features: Number of features per sample
        num_per_block: Number of training samples to load into each block
        train_path: Path to the train file (binary)

    Returns:
        Path to the file containing shuffled data
    """
    assert algorithm in ALGORITHMS, 'Invalid shuffling algorithm provided.'
    if algorithm == 'external_sort':
        return external_sort(dtype, n, num_per_block, num_features, train_path)
    return external_shuffle(dtype, n, num_per_block, num_features, train_path)


def external_sort(
        dtype: str,
        n: int,
        num_features: int,
        num_per_block: int,
        train_path: str) -> None:
    """Shuffle the file using external sort.

    We use an external sorting algorithm by first assigning random indices to
    each sample. Then, sort by indices, where the external merge algorithm takes
    O(nlogn) time.

    Take each block of N samples, and sort in-memory. Save each block to disk.
    For each of k blocks, buffer N/k samples. Then perform a (k-1)-way merge
    using k file buffers, where we load the next N/k samples any time a buffer
    empties.

    Args:
        dtype: Data type of samples in file
        n: Number of total samples
        num_features: Number of features per sample
        num_per_block: Number of training samples to load into each block
        train_path: Path to the train file (binary)
    """
    raise NotImplementedError


def external_shuffle(
        dtype: str,
        n: int,
        num_features: int,
        num_per_block: int,
        train_path: str) -> str:
    """Shuffle the data, and save the shuffled data.

    Args:
        dtype: Data type of samples in file
        n: Number of total samples
        num_features: Number of features per sample
        num_per_block: Number of training samples to load into each block
        train_path: Path to the train file (binary)

    Returns:
        Path to the file containing shuffled data
    """
    num_buffers = int(np.ceil(n / num_per_block))
    shuffled_train_path = train_path + '.tmp'
    with BlockScope(dtype, 'samplesort', num_per_block) as scope:
        writer = BlockWriter(
            dtype,
            num_per_block,
            num_features + 1,
            shuffled_train_path)
        blocks = BlockBuffer(dtype, n, num_features + 1, num_per_block, train_path)
        buffers = []

        for i, block in enumerate(blocks):
            np.random.shuffle(block)
            scope.write_block(i, block)
            buffer = scope.get_block_buffer(i, num_per_block, num_features + 1,
                                            num_per_block // num_buffers)
            buffers.append(buffer)

        while buffers:
            current_block, remaining_buffers = None, []
            for i, buffer in enumerate(buffers[:]):
                try:
                    block = next(buffer)
                except StopIteration:
                    continue
                remaining_buffers.append(buffer)
                if current_block is None:
                    current_block = block
                else:
                    current_block = np.concatenate((current_block, block))
            buffers = remaining_buffers
            if current_block is not None:
                np.random.shuffle(current_block)
                writer.write(current_block)
    return shuffled_train_path
