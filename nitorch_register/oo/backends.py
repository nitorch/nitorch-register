from .imports import MissingImport
import torch
from torch.utils.dlpack import (
    to_dlpack as torch_to_dlpack,
    from_dlpack as torch_from_dlpack,
)

try:
    import numpy as np
except ImportError as error:
    np = MissingImport(error)
try:
    import cupy as cp
    try:
        from cupy import from_dlpack as cupy_from_dlpack
    except ImportError:
        from cupy import fromDlpack as cupy_from_dlpack
    try:
        from cupy import to_dlpack as cupy_to_dlpack
    except ImportError:
        import cupy
        cupy_to_dlpack = cupy.ndarray.toDlpack
except ImportError as error:
    cp = MissingImport(error)


def is_cupy(x):
    if isinstance(cp, MissingImport):
        return False
    return isinstance(x, cp.ndarray)


def is_numpy(x):
    if isinstance(np, MissingImport):
        return False
    return isinstance(x, np.ndarray)


def to_cupy(x):
    """Convert a torch tensor to cupy without copy"""
    return cupy_from_dlpack(torch_to_dlpack(x))


def from_cupy(x):
    """Convert a cupy tensor to torch without copy"""
    return torch_from_dlpack(cupy_to_dlpack(x))


def to_numpy(x):
    """Convert a torch tensor to numpy without copy"""
    return x.numpy()


def from_numpy(x):
    """Convert a numpy tensor to torch without copy"""
    return torch.as_tensor(x)
