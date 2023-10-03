import torch
import os
import nitorch_io as io
from nitorch_core.py import fileparts
from nitorch_core.optionals import numpy as np
from nitorch_register.affine import affine_reorient, volume_layout_to_name


def movedim(x, source, target):
    if np and isinstance(x, np.ndarray):
        return np.moveaxis(x, source, target)
    else:
        return x.movedim(source, target)


def ndim(x):
    if np and isinstance(x, np.ndarray):
        return x.ndim
    elif torch.is_tensor(x):
        return x.dim()
    else:  # MappedArray?
        return x.dim


def reorient(inp, layout='RAS', output=None, transform=None):
    """Shuffle the data to match a given orientation.

    Parameters
    ----------
    inp : str or (tensor, tensor)
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    layout : str or layout-like, default='RAS'
        Target orientation.
    output : str, optional
        Output filename.
        If the input is not a path, the reoriented data is not written
        on disk by default.
        If the input is a path, the default output filename is
        '{dir}/{base}.{layout}{ext}', where `dir`, `base` and `ext`
        are the directory, base name and extension of the input file.
    transform : str, optional
        Output filename of the corresponding transform.
        Not written by default.

    Returns
    -------
    output : str or (tensor, tensor)
        If the input is a path, the output path is returned.
        Else, the reoriented data and orientation matrix are returned.

    """
    dir = ''
    base = ''
    ext = ''
    fname = ''

    is_file = isinstance(inp, str)
    if is_file:
        fname = inp
        f = io.volumes.map(inp)
        try:
            inp = (f.data(), f.affine)
        except Exception:
            inp = (f.data(dtype=torch.long), f.affine)
        if output is None:
            output = '{dir}{sep}{base}.{layout}{ext}'
        dir, base, ext = fileparts(fname)

    dat, aff0 = inp
    dim = aff0.shape[-1] - 1
    datdim = ndim(dat)
    dat = movedim(dat, range(dim, datdim), range(datdim-dim))
    aff, dat = affine_reorient(aff0, dat, layout)
    dat = movedim(dat, range(datdim-dim), range(dim, datdim))

    if isinstance(dat, io.MappedArray):
        dat = dat.data(numpy=True)

    if output:
        layout = volume_layout_to_name(layout)
        if is_file:
            output = output.format(dir=dir or '.', base=base, ext=ext,
                                   sep=os.path.sep, layout=layout)
            io.volumes.save(dat, output, like=fname, affine=aff)
        else:
            output = output.format(sep=os.path.sep, layout=layout)
            io.volumes.save(dat, output, affine=aff)

    if transform:
        if is_file:
            transform = transform.format(dir=dir or '.', base=base, ext=ext,
                                         sep=os.path.sep, layout=layout)
        else:
            transform = transform.format(sep=os.path.sep, layout=layout)
        io.transforms.savef(torch.eye(4), transform, source=aff0, target=aff)

    if is_file:
        return output
    else:
        return dat, aff
