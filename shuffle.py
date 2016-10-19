"""Shuffling algorithm implementations for SSGD.

All of the following shuffling algorithms take advantage of spatial locality
by loading data from and writing data to disk sequentially, wherever possible.
"""

import numpy as np

from blocks import BlockBuffer
from blocks import BlockScope
from blocks import BlockWriter

ALGORITHMS = ('external_sort', 'external_shuffle')


def shuffle_train(
        algorithm: str,
        dtype: str,
        n: int,
        num_per_block: int,
        train_path: str) -> None:
    """Invoke the correct shuffling algorithm."""
    assert algorithm in ALGORITHMS, 'Invalid shuffling algorithm provided.'
    if algorithm == 'external_sort':
        external_sort(dtype, n, num_per_block, train_path)
    else:
        external_shuffle(dtype, n, num_per_block, train_path)


def external_sort(
        dtype: str,
        n: int,
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
        num_per_block: Number of training samples to load into each block
        train_path: Path to the train file (binary)
    """
    pass


def external_shuffle(
        dtype: str,
        n: int,
        num_per_block: int,
        train_path: str) -> None:
    """Shuffle the data, and save the shuffled data.

    *Note* May cause issues if num_per_block is not evenly divided by the
    number of buffers. This means some entry in the block will be left empty.

    Args:
        dtype: Data type of samples in file
        n: Number of total samples
        num_per_block: Number of training samples to load into each block
        train_path: Path to the train file (binary)
    """
    num_buffers = n / num_per_block
    with BlockScope('float64', 'samplesort', num_per_block) as scope:
        writer = BlockWriter(dtype, num_per_block, train_path)
        blocks = BlockBuffer(dtype, num_per_block, train_path)
        buffers = []

        for i, block in enumerate(blocks):
            scope.write_block(i, np.random.shuffle(block))
            buffer = scope.get_block_buffer(i, num_per_block // num_buffers)
            buffers.append(buffer)

        while buffers:
            current_block = np.matrix([])
            for i, buffer in enumerate(buffers[:]):
                block = buffer.read_block()
                current_block = np.concatenate((current_block, block))
                if len(block) == 0:
                    buffers.pop(i)
            np.random.shuffle(current_block)
            writer.write(current_block)
