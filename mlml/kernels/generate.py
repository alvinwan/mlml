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
from mlml.logging import timeit
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
            data: Data=None,
            memId: str=time.time(),
            dir: str= './'):
        self.dtype = dtype
        self.function = function
        self.num_samples = num_samples
        self.memId = memId
        self.kernel_path = os.path.join(dir, MemKernel.PATH.format(id=memId))
        self.n, self.d = data.X.shape
        self.data = data

    @timeit
    def generate(self):
        """Generate kernel from matrix X and save to disk."""
        assert self.data is not None, 'Data required to generate kernel matrix.'
        print(' * [MemKernel] Generating kernel matrix', self.memId,
              '(', self.dtype, ')')
        s, rows_written, cols_written = min(self.num_samples, self.n), 0, 0
        writer = BlockWriter(self.dtype, self.n, s, self.kernel_path)
        for i in range(ceil(self.n / s)):
            partial = None
            Xi = self.data.X[i * s: (i + 1) * s]
            rows_written += Xi.shape[0]
            yi = self.data.labels[i * s: (i + 1) * s]
            yi.shape = (yi.shape[0], 1)
            for j in range(ceil(self.n / s)):
                Xj = self.data.X[j * s: (j + 1) * s]
                K_ij = self.function(Xi, Xj)
                partial = K_ij if partial is None else \
                    np.concatenate((partial, K_ij), axis=1)
            partial = np.concatenate((partial, yi), axis=1)
            cols_written = partial.shape[1]
            print(' * [MemKernel] Generated partial', i)
            writer.write(partial)
            print(' * [MemKernel] Wrote partial', i)
        print(' * [MemKernel] Finished', (rows_written, cols_written),
              'Kernel', self.memId)
        return self

    @timeit
    def generate_rbf(self, simulated: bool=False):
        """Generate an RBF kernel more efficiently, if simulated."""
        if simulated:
            mmap = np.memmap(self.kernel_path, self.dtype, mode='w+', shape=(self.n, self.n))
            mmap[:] = self.function(self.data.X, self.data.X)
            del mmap

            print(' * [MemKernel] Finished Kernel', self.memId)
            return self
        return self.generate()


class RidgeRegressionKernel(MemKernel):
    """Kernel precomputation specific to Ridge Regression."""

    LAMBDA_PATH = MemKernel.PATH_PREFIX + '-Lambda.tmp'

    def __init__(
            self,
            function: Callable[[np.ndarray, np.ndarray], np.ndarray],
            num_samples: int,
            data: Data,
            dir: str = './',
            dtype: str = 'float16',
            mem_id: str=time.time(),
            reg: float = 0.1):
        super(RidgeRegressionKernel, self).__init__(
            dtype, function, num_samples, data, mem_id, dir)
        self.Lambda_path = os.path.join(dir, self.LAMBDA_PATH.format(id=mem_id))
        self.Lambda = MemMatrix(
            num_samples, self.Lambda_path, self.dtype, mode='r+',
            shape=(self.n, self.n)) if os.path.exists(self.Lambda_path) else None
        self.reg = reg

    @timeit
    def generate_Lambda(self):
        """Generate the matrix K + lambda I."""
        copyfile(self.kernel_path, self.Lambda_path)
        print(' * [MemKernel] Generating Lambda')
        n = self.data.X.shape[0]
        mode = 'r+' if os.path.exists(self.Lambda_path) else 'w+'
        fh = np.memmap(self.Lambda_path, self.dtype, mode=mode, shape=(n, n + 1))
        for i in range(self.data.X.shape[0]):
            fh[i, i] += self.reg
        self.Lambda = MemMatrix(self.num_samples, self.Lambda_path, self.dtype,
                            mode='r+', shape=(n, n))
        del fh
        return self

