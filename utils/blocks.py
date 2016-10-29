"""Utilities for loading and writing blocks of data to disk.

These utilities are mostly used by the permutation algorithms, which perform
the most complex maneuvers for I/O. SSGD itself also uses selective-read
utilities from this module.
"""

import os
import numpy as np


def bytes_per_dtype(dtype: str) -> int:
    """Compute number of bytes for this dtype."""
    suffixes = (('8', 1), ('16', 2), ('32', 4), ('64', 8), ('_', 8))
    for suffix, size in suffixes:
        if dtype.endswith(suffix):
            return size


class BlockBuffer:
    """File buffer that buffers blocks of data at once.

    To use, initialize and iterate over buffer. Blocks will be yielded one at
    a time, until there are no more blocks to read from the relevant file.

        buffer = BlockBuffer('float64', 1024, 'path/to/file')
        for block in buffer:
            ...

    Note that the file is never completely stored in memory.
    """

    def __init__(
            self,
            dtype: str,
            n: int,
            num_entries: int,
            num_per_block: int,
            path: str):
        """Initialize file handler but do not buffer data.

        Args:
            dtype: Data type of numbers in file
            n: Number of samples in total
            num_entries: Number of entries per row
            num_per_block: Number of rows per block
            path: Path to the file to buffer
        """
        self.block = 0
        self.bytes_per_entry = bytes_per_dtype(dtype) * num_entries
        self.dtype = dtype
        self.n = n
        self.num_entries = num_entries
        self.num_per_block = int(num_per_block)
        self.path = path

    def __next__(self) -> np.ndarray:
        """Buffer and return the next block of data.

        Returns:
            The next buffered block of data, as a numpy matrix
        """
        block = self.read_block(self.block)
        self.block += 1
        if len(block) == 0:
            raise StopIteration
        return block

    def read_block(self, block: int) -> np.ndarray:
        """Read block of data from shuffled data.

        Note that even though the entire I/O buffer is run through, only data
        from the current block is saved in memory and returned to the main sgd
        loop for training.

        Args:
            block: Index of the block of data to read into memory

        Returns:
            A tuple containing training inputs and outputs
        """
        remainder = max(0, self.n - (block * self.num_per_block))
        num = min(self.num_per_block, remainder)
        return np.matrix(np.memmap(
            self.path,
            dtype=self.dtype,
            mode='r+',
            offset=block * (self.bytes_per_entry * self.num_per_block),
            shape=(num, self.num_entries)))

    def __iter__(self):
        return self


class BlockScope:
    """Handles blocks of data in temporary files.

    The BlockScope will only write to and read from temporary files in the
    current BlockScope. To use, use a with statement to create a new scope.

        with BlockScope('float64', 'test', 1024) as scope:
            ...
            scope.write_block(block_id, data)
            ...
            buffer = get_buffer(block_id)

    Once the scope has closed, all temporary files will be deleted.
    """

    BLOCK_SCOPE_FILENAME_FORMAT = 'data/{namespace}-{id}.tmp'

    def __init__(self, dtype: str, namespace: str, num_per_block: int):
        """Initialize file handler but do not buffer data.

        Args:
            dtype: Data type of numbers in file
            namespace: Prefix for all temporary files
            num_per_block: Number of samples per block
        """
        self.block = 0
        self.dtype = dtype
        self.namespace = namespace
        self.num_per_block = num_per_block
        self.paths = []

    def __enter__(self):
        """Initialize the set of temporary files."""
        return self

    def __exit__(self, *args):
        """Destroy the set of temporary files."""
        for path in self.paths:
            if os.path.exists(path):
                os.remove(path)

    def write_block(self, block_id: int, data: np.ndarray) -> None:
        """Writes a block of data to a temporary file in this scope.

        Args:
            block_id: Unique id for the block to write to
            data: The data to write into the file
        """
        path = BlockScope.BLOCK_SCOPE_FILENAME_FORMAT.format(
            namespace=self.namespace,
            id=block_id)
        writer = BlockWriter(
            self.dtype,
            data.shape[0],
            self.num_per_block,
            path)
        writer.offset = block_id
        writer.write(data)

    def get_block_buffer(
            self,
            block_id: int,
            n: int,
            num_entries: int,
            num_per_block: int) -> BlockBuffer:
        """Reads a block of data from a temporary file in this scope.

        Args:
            block_id: Unique id for the block to read from
            n: Number of samples
            num_entries: Number of entries in each row
            num_per_block: Number per block for new buffer to give
        """
        path = BlockScope.BLOCK_SCOPE_FILENAME_FORMAT.format(
            namespace=self.namespace,
            id=block_id)
        assert os.path.exists(path), 'File not found: %s' % path
        return BlockBuffer(self.dtype, n, num_entries, num_per_block, path)


class BlockWriter:
    """Writes to block within a numpy binary file."""

    def __init__(
            self,
            dtype: str,
            num_entries: int,
            num_per_block: int,
            path: str):
        """Initialize the block writer

        Args:
            dtype: Type of data to write and read
            num_entries: Number of entries per row
            num_per_block: The number of rows per block
            path: Path to write to
        """
        self.bytes_per_entry = bytes_per_dtype(dtype) * num_entries
        self.dtype = dtype
        self.num_per_block = num_per_block
        self.offset = 0
        self.path = path

    def write(self, data: np.ndarray, mode: str='r+'):
        """Write a block of data to disk.

        Args:
            data: A block of data
            mode: The memmap mode
        """
        if not os.path.exists(self.path):
            mode = 'w+'
        handler = np.memmap(
            self.path,
            dtype=self.dtype,
            mode=mode,
            offset=self.offset * (self.bytes_per_entry * self.num_per_block),
            shape=data.shape)
        handler[:] = data[:]
        self.offset += 1
        del handler
