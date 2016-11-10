"""Utilities for generating a kernel matrix, given a kernel function.

For the following, we note that our original optimization problem has the
following solution, given a ridge regression objective function, a kernel matrix
K and a kernel function k.

X^T(K + lambda I)

"""

import numpy as np
import os
import time

from math import ceil
from mlml.ssgd.blocks import BlockBuffer
from mlml.ssgd.blocks import BlockWriter
from mlml.ssgd.blocks import bytes_per_dtype
from mlml.utils.data import Data
from shutil import copyfile
from typing import Callable
from typing import Union


class MemMatrix:
    """Representation of a matrix stored on disk.

    Memory-mapped files in numpy provide memory-efficient, lazy load
    functionality, but this utility builds on that to provide memory-optimized
    computation.
    """

    def __init__(
            self,
            num_samples: int,
            filename: str,
            dtype: str='uint8',
            mode: str='r+',
            offset: int=0,
            shape: tuple=None,
            order: str='C'):
        self.dtype = dtype
        self.filename = filename
        self.n, self.d = shape
        self.mode = mode
        self.offset = offset
        self.shape = shape
        self.num_samples = num_samples
        self.offset = offset
        self.order = order

    def dot(
            self,
            M: Union[np.ndarray, np.memmap, 'MemMatrix'],
            mmap: bool=False,
            path: str=None):
        """Compute dot product between this memory-mapped matrix and M.

        This function handles all

        Args:
            M: Matrix to compute dot products with, either on disk or in memory
            mmap: Returns a memory-mapped file if true.

        Returns:
            A memory-mapped file or an in-memory matrix.
        """
        assert not (mmap is True and path is None), 'Path must be given.'
        assert self.shape[1] == M.shape[0], 'Shape mismatch.'
        is_mmap = isinstance(M, (np.memmap, MemMatrix))
        if mmap and not is_mmap:
            raise NotImplementedError
        elif mmap and is_mmap:
            n, s = self.n, min(self.num_samples // 2, self.n)
            reader = np.memmap(self.filename, self.dtype, mode='r', shape=(n, n))
            indices_per_dim = ceil(n / s)
            for j in range(indices_per_dim):
                cols = reader[:, j * s: (j + 1) * s]
                for i in range(indices_per_dim):
                    offset = bytes_per_dtype(self.dtype) * s * \
                             (j % indices_per_dim) * (i * indices_per_dim)
                    mode = 'r+' if os.path.exists(path) else 'w+'
                    writer = np.memmap(path, self.dtype, mode=mode,
                                       shape=(s, s), offset=offset)
                    rows = reader[i * s: (i + 1) * s]
                    writer[:] = rows.dot(cols)
            return MemMatrix(self.num_samples, path, self.dtype, mode='r+',
                             shape=(n, M.d))
        elif not mmap and not is_mmap:
            buffer = BlockBuffer(
                self.dtype, self.n, self.d, self.num_samples, self.filename)
            return np.concatenate(tuple(block.dot(M) for block in buffer))
        else:
            raise NotImplementedError


class MemKernel:
    """Generates and saves kernel matrix to disk.

    This operates off of the assumption that X can fully fit in memory.
    """

    # Path prefix for Kernel files
    # Param: id - unique identifier for the generated kernel
    PATH_PREFIX = 'mem-{id}'

    PATH = PATH_PREFIX + '-kernel.tmp'

    def __init__(
            self,
            dtype: str,
            function: Callable[[np.ndarray, np.ndarray], np.ndarray],
            num_samples: int,
            data: Data,
            memId: str=time.time(),
            dir: str= './'):
        self.dtype = dtype
        self.function = function
        self.num_samples = num_samples
        self.memId = memId
        self.kernel_path = os.path.join(dir, MemKernel.PATH.format(id=memId))
        self.n, self.d = data.X.shape
        self.data = data

    def generate(self):
        """Generate kernel from matrix X and save to disk."""
        print(' * [MemKernel] Generating kernel matrix', self.memId)
        s, rows_written = self.num_samples, 0
        writer = BlockWriter(self.dtype, self.n, s, self.kernel_path)
        for o in range(ceil(self.n / s)):
            partial = np.zeros((s, self.n + 1))
            for i, row in enumerate(self.data.X[o * s: (o + 1) * s]):
                for j, col in enumerate(self.data.X):
                    partial[i][j] = self.function(row, col)
                partial[i][self.n] = self.data.labels[i]
                rows_written += 1
            writer.write(partial)
            print(' * [MemKernel] Wrote partial', o)
        print(' * [MemKernel] Finished', (rows_written, self.n + 1),
              'Kernel', self.memId)
        return self


class RidgeRegressionKernel(MemKernel):
    """Kernel precomputation specific to Ridge Regression."""

    A1_PATH = MemKernel.PATH_PREFIX + '-A1.tmp'
    A2_PATH = MemKernel.PATH_PREFIX + '-A2.tmp'

    def __init__(
            self,
            dtype: str,
            function: Callable[[np.ndarray, np.ndarray], np.ndarray],
            num_samples: int,
            data: Data,
            memId: str=time.time(),
            reg: float = 0.1,
            dir: str = './'):
        super(RidgeRegressionKernel, self).__init__(
            dtype, function, num_samples, data, memId, dir)
        self.a1_path = os.path.join(dir, self.A1_PATH.format(id=memId))
        self.a2_path = os.path.join(dir, self.A2_PATH.format(id=memId))
        self.A1 = MemMatrix(
            num_samples, self.a1_path, self.dtype, mode='r+',
            shape=(self.n, self.n)) if os.path.exists(self.a1_path) else None
        self.A2 = MemMatrix(
            num_samples, self.a2_path, self.dtype, mode='r+',
            shape=(self.n, self.n)) if os.path.exists(self.a2_path) else None
        self.reg = reg

    def generate_A1(self):
        """Generate the matrix K + lambda I."""
        copyfile(self.kernel_path, self.a1_path)
        print(' * [MemKernel] Generating A1')
        n = self.data.X.shape[0]
        mode = 'r+' if os.path.exists(self.a1_path) else 'w+'
        fh = np.memmap(self.a1_path, self.dtype, mode=mode, shape=(n, n))
        for i in range(self.data.X.shape[0]):
            fh[i, i] += self.reg
        self.A1 = MemMatrix(self.num_samples, self.a1_path, self.dtype,
                            mode='r+', shape=(n, n))
        del fh
        return self

    def generate_A2(self):
        """Generate the matrix A1^T A1 = A1 A1"""
        print(' * [MemKernel] Generating A2')
        self.A2 = self.A1.dot(self.A1, mmap=True, path=self.a2_path)
        return self

    def generate_A3(self):
        """Generate the matrix A1 y."""
        self.A3 = self.A1.dot(self.data.Y)
        return self

    def gradient(self, M: np.ndarray):
        """Evaluate gradient using A2 * B - A3."""
        return self.A2.dot(M) - self.A3
