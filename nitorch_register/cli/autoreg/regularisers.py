"""
This file implements spatial regularizers for vector and displacement fields.

All functions exist in a "field" (by default) for vector fields or "grid"
for displacement fields.

- absolute(x) / absolute_grid(v)
    Squared norm of absolute values
- membrane(x) / membrane_grid(v)
    Squared norm of first order derivatives
- bending(x) / bending_grid(v)
    Squared norm of second order derivatives
- lame_div(v) [displacement only]
    Squared norm of the trace of the Jacobian matrix
- lame_shear(v) [displacement only]
    Squared norm of the off-diagonal elements of the Jacobian matrix
- regulariser(x) / regulariser_grid(v)
    Combination of all regularisers.
All these functions accept non-stationary weight maps.

Another set of functions generate an equivalent convolution kernel
- regulariser_kernel / regulariser_grid_kernel
    Return a sparse convolution kernel

A set of functions return the diagonal of each regulariser (when
though of in matrix form). They are only implemented in the "field" case.
- absolute_diag
- membrane_diag
- bending_diag

Finally, some functions compute updated weight maps in the RLS context:
- membrane_weights
- bending_weights
These functions are not yet implemented for the other regularisers (WIP).
"""

import torch
import itertools
from nitorch_core.extra import (
    make_vector, fast_movedim, backend as get_backend)
from nitorch_core.py import ensure_list
from nitorch_core.finite_differences import diff, div, diff1d, div1d


def _mul_(x, y):
    """Smart in-place multiplication"""
    if torch.is_tensor(y) and y.requires_grad:
        return x * y
    else:
        return x.mul_(y)


def _div_(x, y):
    """Smart in-place division"""
    if torch.is_tensor(y) and y.requires_grad:
        return x / y
    else:
        return x.div_(y)


def absolute(field, weights=None):
    """Precision matrix for the Absolute energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : tensor

    """
    if weights is not None:
        backend = dict(dtype=field.dtype, device=field.device)
        weights = torch.as_tensor(weights, **backend)
        return field * weights
    else:
        return field


def absolute_grid(grid, voxel_size=1, weights=None):
    """Precision matrix for the Absolute energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    grid = grid * voxel_size.square()
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)
        if weights.requires_grad:
            grid = grid * weights[..., None]
        else:
            grid *= weights[..., None]
    return grid


def membrane(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Membrane energy

    Note
    ----
    .. This is exactly equivalent to SPM's membrane energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'
    dim : int, default=field.dim()
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial) tensor

    """
    if weights is None:
        return _membrane_l2(field, voxel_size, bound, dim)

    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    if torch.is_tensor(voxel_size):
        voxel_size = make_vector(voxel_size, dim, **backend)
    else:
        voxel_size = ensure_list(voxel_size, dim)
    bound = ensure_list(bound, dim)
    dims = list(range(field.dim() - dim, field.dim()))
    weights = torch.as_tensor(weights, **backend)

    # dims = list(range(field.dim()-dim, field.dim()))
    # fieldb = diff(field, dim=dims, voxel_size=voxel_size, side='b', bound=bound)
    # fieldb = mul_(fieldb, weights[..., None])
    # dims = list(range(fieldb.dim() - 1 - dim, fieldb.dim() - 1))
    # fieldb = div(fieldb, dim=dims, voxel_size=voxel_size, side='b', bound=bound)
    #
    # dims = list(range(field.dim()-dim, field.dim()))
    # field = diff(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    # field = mul_(field, weights[..., None])
    # dims = list(range(field.dim()-1-dim, field.dim()-1))
    # field = div(field, dim=dims, voxel_size=voxel_size, side='f', bound=bound)
    # field += fieldb
    # field *= 0.5

    if not field.requires_grad:
        buf1 = torch.empty_like(field)
        buf2 = torch.empty_like(field)
    else:
        buf1 = buf2 = None

    mom = 0
    for i in range(dim):
        for side in ('f', 'b'):
            g = diff1d(field, dim=dims[i], bound=bound[i],
                       voxel_size=voxel_size[i], side=side, out=buf1)
            g = _mul_(g, weights)
            mom += div1d(g, dim=dims[i], bound=bound[i],
                         voxel_size=voxel_size[i], side=side, out=buf2)

    mom *= 0.5
    return mom


def _membrane_l2(field, voxel_size=1, bound='dct2', dim=None):
    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    if torch.is_tensor(voxel_size):
        voxel_size = make_vector(voxel_size, dim, **backend)

    dims = list(range(field.dim() - dim, field.dim()))
    fieldf = diff(field, dim=dims, voxel_size=voxel_size, side='f',
                  bound=bound)
    dims = list(range(fieldf.dim() - 1 - dim, fieldf.dim() - 1))
    field = div(fieldf, dim=dims, voxel_size=voxel_size, side='f', bound=bound)

    return field


def membrane_grid(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Membrane energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    grid = fast_movedim(grid, -1, -(dim + 1))
    grid = membrane(grid, weights=weights, voxel_size=voxel_size,
                    bound=bound, dim=dim)
    grid = fast_movedim(grid, -(dim + 1), -1)
    if (voxel_size != 1).any():
        grid.mul_(voxel_size.square())
    return grid


def bending(field, voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for the Bending energy

    Note
    ----
    .. This is exactly equivalent to SPM's bending energy

    Parameters
    ----------
    field : (..., *spatial) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dct2'
    dim : int, default=field.dim()
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial) tensor

    """
    if weights is None:
        return _bending_l2(field, voxel_size, bound, dim)

    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    if torch.is_tensor(voxel_size):
        voxel_size = make_vector(voxel_size, dim, **backend)
    else:
        voxel_size = ensure_list(voxel_size, dim)
    bound = ensure_list(bound, dim)
    dims = list(range(field.dim() - dim, field.dim()))
    weights = torch.as_tensor(weights, **backend)

    if not field.requires_grad:
        bufi = torch.empty_like(field)
        bufij = torch.empty_like(field)
        bufjj = torch.empty_like(field)
    else:
        bufi = bufij = bufjj = None

    mom = 0
    for i in range(dim):
        for side_i in ('f', 'b'):
            opti = dict(dim=dims[i], bound=bound[i], side=side_i,
                        voxel_size=voxel_size[i])
            di = diff1d(field, **opti, out=bufi)
            for j in range(i, dim):
                for side_j in ('f', 'b'):
                    optj = dict(dim=dims[j], bound=bound[j], side=side_j,
                                voxel_size=voxel_size[j])
                    dj = diff1d(di, **optj, out=bufij)
                    dj = _mul_(dj, weights)
                    dj = div1d(dj, **optj, out=bufjj)
                    dj = div1d(dj, **opti, out=bufij)
                    if i != j:
                        # off diagonal -> x2  (upper + lower element)
                        dj.mul_(2)
                    mom += dj
    mom.div_(4.)
    return mom


def _bending_l2(field, voxel_size=1, bound='dct2', dim=None):
    backend = dict(dtype=field.dtype, device=field.device)
    dim = dim or field.dim()
    if torch.is_tensor(voxel_size):
        voxel_size = make_vector(voxel_size, dim, **backend)
    else:
        voxel_size = ensure_list(voxel_size, dim)
    bound = ensure_list(bound, dim)
    dims = list(range(field.dim() - dim, field.dim()))

    # allocate buffers
    if not field.requires_grad:
        bufi = torch.empty_like(field)
        bufij = torch.empty_like(field)
        bufjj = torch.empty_like(field)
    else:
        bufi = bufij = bufjj = None

    mom = 0
    for i in range(dim):
        for side_i in ('f', 'b'):
            opti = dict(dim=dims[i], bound=bound[i], side=side_i,
                        voxel_size=voxel_size[i])
            di = diff1d(field, **opti, out=bufi)
            for j in range(i, dim):
                for side_j in ('f', 'b'):
                    optj = dict(dim=dims[j], bound=bound[j], side=side_j,
                                voxel_size=voxel_size[j])
                    dj = diff1d(di, **optj, out=bufij)
                    dj = div1d(dj, **optj, out=bufjj)
                    dj = div1d(dj, **opti, out=bufij)
                    if i != j:
                        # off diagonal -> x2  (upper + lower element)
                        dj = dj.mul_(2)
                    mom += dj
    mom = mom.div_(4.)
    return mom


def bending_grid(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Bending energy of a deformation grid

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    field : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    grid = fast_movedim(grid, -1, -(dim + 1))
    grid = bending(grid, weights=weights, voxel_size=voxel_size,
                   bound=bound, dim=dim)
    grid = fast_movedim(grid, -(dim + 1), -1)
    if (voxel_size != 1).any():
        grid.mul_(voxel_size.square())
    return grid


def lame_shear(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Shear component of the Linear-Elastic energy.

    Notes
    -----
    .. This regulariser can only be applied to deformation fields.
    .. It corresponds to the second Lame constant (the shear modulus).
    .. This is exactly equivalent to SPM's linear-elastic energy.
    .. It penalizes the Frobenius norm of the symmetric part
       of the Jacobian (shears on the off-diagonal and zooms on the
       diagonal).
    .. Formaly: `<f, Lf> = \int \sum_i (df_i/dx_i)^2
                                + \sum_{j > i} (df_j/dx_i + df_i/dx_j)^2 dx

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    if weights is None:
        return _lame_shear_l2(grid, voxel_size, bound)

    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    bound = ensure_list(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)

    mom = torch.zeros_like(grid)
    for i in range(dim):
        # symmetric part
        x_i = grid[..., i]
        for j in range(i, dim):
            for side_i in ('f', 'b'):
                opt_ij = dict(dim=dims[j], side=side_i, bound=bound[j],
                              voxel_size=voxel_size[j])
                diff_ij = diff1d(x_i, **opt_ij).mul_(voxel_size[i])
                if i == j:
                    # diagonal elements
                    diff_ij_w = diff_ij if weights is None else diff_ij * weights
                    mom[..., i].add_(div1d(diff_ij_w, **opt_ij), alpha=0.5)
                else:
                    # off diagonal elements
                    x_j = grid[..., j]
                    for side_j in ('f', 'b'):
                        opt_ji = dict(dim=dims[i], side=side_j, bound=bound[i],
                                      voxel_size=voxel_size[i])
                        diff_ji = diff1d(x_j, **opt_ji).mul_(voxel_size[j])
                        diff_ji = diff_ji.add_(diff_ij).mul_(0.5)
                        if weights is not None:
                            diff_ji = diff_ji * weights
                        mom[..., j].add_(div1d(diff_ji, **opt_ji), alpha=0.25)
                        mom[..., i].add_(div1d(diff_ji, **opt_ij), alpha=0.25)
                    del x_j
        del x_i
    del grid

    mom *= 2 * voxel_size  # JA added an additional factor 2 to the kernel
    return mom


def _lame_shear_l2(grid, voxel_size=1, bound='dft'):
    backend = dict(dtype=grid.dtype, device=grid.device)
    dim = grid.shape[-1]
    voxel_size = make_vector(voxel_size, dim, **backend)
    bound = ensure_list(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))

    # allocate buffers
    if not grid.requires_grad:
        buf1 = torch.empty_like(grid[..., 0])
        buf2 = torch.empty_like(buf1)
        buf3 = torch.empty_like(buf1)
    else:
        buf1 = buf2 = buf3 = None

    mom = torch.zeros_like(grid)
    for i in range(dim):
        # symmetric part
        x_i = grid[..., i]
        for j in range(i, dim):
            for side_i in ('f', 'b'):
                opt_ij = dict(dim=dims[j], side=side_i, bound=bound[j],
                              voxel_size=voxel_size[j])
                diff_ij = diff1d(x_i, **opt_ij, out=buf1).mul_(voxel_size[i])
                if i == j:
                    # diagonal elements
                    mom[..., i].add_(div1d(diff_ij, **opt_ij, out=buf2),
                                     alpha=0.5)
                else:
                    # off diagonal elements
                    x_j = grid[..., j]
                    for side_j in ('f', 'b'):
                        opt_ji = dict(dim=dims[i], side=side_j, bound=bound[i],
                                      voxel_size=voxel_size[i])
                        diff_ji = diff1d(x_j, **opt_ji, out=buf2).mul_(
                            voxel_size[j])
                        diff_ji.add_(diff_ij, alpha=0.5)
                        mom[..., j].add_(div1d(diff_ji, **opt_ji, out=buf3),
                                         alpha=0.25)
                        mom[..., i].add_(div1d(diff_ji, **opt_ij, out=buf3),
                                         alpha=0.25)
                    del x_j
        del x_i
    del grid

    mom.mul_(2 * voxel_size)  # JA added an additional factor 2 to the kernel
    return mom


def lame_div(grid, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for the Divergence component of the Linear-Elastic energy.

    Notes
    -----
    .. This regulariser can only be applied to deformation fields.
    .. It corresponds to the first Lame constant (the divergence).
    .. This is exactly equivalent to SPM's linear-elastic energy.
    .. It penalizes the square of the trace of the Jacobian
       (i.e., volume changes)
    .. Formally: `<f, Lf> = \int [(\sum_i df_i/dx_i)^2 - \sum_i (df_i/dx_i)^2] dx

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
    voxel_size : float or sequence[float], default=1 (actually unused)
    bound : str, default='dft'
    weights : (..., *spatial) tensor, optional

    Returns
    -------
    grid : (..., *spatial, dim) tensor

    """
    if weights is None:
        return _lame_div_l2(grid, voxel_size, bound)

    dim = grid.shape[-1]
    bound = ensure_list(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))
    if weights is not None:
        backend = dict(dtype=grid.dtype, device=grid.device)
        weights = torch.as_tensor(weights, **backend)

    # precompute gradients
    grad = [dict(f={}, b={}) for _ in range(dim)]
    opt = [dict(f={}, b={}) for _ in range(dim)]
    for i in range(dim):
        x_i = grid[..., i]
        for side in ('f', 'b'):
            opt_i = dict(dim=dims[i], side=side, bound=bound[i])
            grad[i][side] = diff1d(x_i, **opt_i)
            opt[i][side] = opt_i

    # compute divergence
    mom = torch.zeros_like(grid)
    all_sides = list(itertools.product(['f', 'b'], repeat=dim))
    for sides in all_sides:
        div = 0
        for i, side in enumerate(sides):
            div += grad[i][side]
        if weights is not None:
            div = div * weights
        for i, side in enumerate(sides):
            mom[..., i] += div1d(div, **(opt[i][side]))

    mom /= float(2 ** dim)  # weight sides combinations
    return mom


def _lame_div_l2(grid, voxel_size=1, bound='dft'):
    dim = grid.shape[-1]
    bound = ensure_list(bound, dim)
    dims = list(range(grid.dim() - 1 - dim, grid.dim() - 1))

    # precompute gradients
    grad = [dict(f={}, b={}) for _ in range(dim)]
    opt = [dict(f={}, b={}) for _ in range(dim)]
    for i in range(dim):
        x_i = grid[..., i]
        for side in ('f', 'b'):
            opt_i = dict(dim=dims[i], side=side, bound=bound[i])
            grad[i][side] = diff1d(x_i, **opt_i)
            opt[i][side] = opt_i

    if not grid.requires_grad:
        buf1 = torch.empty_like(grid[..., 0])
        buf2 = torch.empty_like(grid[..., 0])
    else:
        buf1 = buf2 = None

    # compute divergence
    mom = torch.zeros_like(grid)
    all_sides = list(itertools.product(['f', 'b'], repeat=dim))
    for sides in all_sides:
        div = buf1.zero_() if buf1 is not None else 0
        for i, side in enumerate(sides):
            div += grad[i][side]
        for i, side in enumerate(sides):
            mom[..., i] += div1d(div, **(opt[i][side]), out=buf2)

    mom /= float(2 ** dim)  # weight sides combinations
    return mom


# aliases to avoid shadowing
_absolute = absolute
_membrane = membrane
_bending = bending


def absolute_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Absolute energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's absolute energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1 (unused)

    Returns
    -------
    kernel : (1,)*dim sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()

    kernel = torch.sparse_coo_tensor(
        torch.zeros([dim, 1], dtype=torch.long, device=device),
        torch.ones([1], dtype=dtype, device=device),
        [1] * dim)
    return kernel


def absolute_grid_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Absolute energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's absolute energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (1,)*dim sparse tensor

    """
    kernel = absolute_kernel(dim, voxel_size, dtype=dtype, device=device)
    voxel_size = make_vector(voxel_size, dim,
                                        **get_backend(kernel))
    kernel = torch.stack([kernel * vx.square() for vx in voxel_size], dim=0)
    return kernel


def membrane_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Membrane energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's membrane energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (3,)*dim sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)
    vx = voxel_size.square().reciprocal()

    # build sparse kernel
    kernel = [2 * vx.sum()]
    center_index = [1] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross
        kernel += [-vx[d]] * 2
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 2
        indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=vx.device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [3] * dim)

    return kernel


def membrane_grid_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Membrane energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's membrane energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (dim, [3,]*dim) sparse tensor

    """
    kernel = membrane_kernel(dim, voxel_size, dtype=dtype, device=device)
    voxel_size = make_vector(voxel_size, dim,
                                        **get_backend(kernel))
    kernel = torch.stack([kernel * vx.square() for vx in voxel_size], dim=0)
    return kernel


def bending_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Bending energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's bending energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (5,)*dim sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)
    vx = voxel_size.square().reciprocal()
    vx2 = vx.square()
    cvx = torch.combinations(vx, r=2).prod(dim=-1)

    # build sparse kernel
    kernel = [6 * vx2.sum() + 8 * cvx.sum()]
    center_index = [2] * dim
    indices = [list(center_index)]
    for d in range(dim):
        # cross 1st order
        kernel += [-4 * vx[d] * vx.sum()] * 2
        index = list(center_index)
        index[d] = 1
        indices.append(index)
        index = list(center_index)
        index[d] = 3
        indices.append(index)
        # cross 2nd order
        kernel += [vx2[d]] * 2
        index = list(center_index)
        index[d] = 0
        indices.append(index)
        index = list(center_index)
        index[d] = 4
        indices.append(index)
        for dd in range(d + 1, dim):
            # off
            kernel += [2 * vx[d] * vx[dd]] * 4
            index = list(center_index)
            index[d] = 1
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 1
            index[dd] = 3
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 1
            indices.append(index)
            index = list(center_index)
            index[d] = 3
            index[dd] = 3
            indices.append(index)
    indices = torch.as_tensor(indices, dtype=torch.long, device=vx.device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel, [5] * dim)

    return kernel


def bending_grid_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Bending energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's bending energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (dim, dim, [5,]*dim) sparse tensor

    """
    kernel = bending_kernel(dim, voxel_size, dtype=dtype, device=device)
    voxel_size = make_vector(voxel_size, dim,
                                        **get_backend(kernel))
    kernel = torch.stack([kernel * vx.square() for vx in voxel_size], dim=0)
    return kernel


def lame_shear_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Linear Elastic energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's LE energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1

    Returns
    -------
    kernel : (dim, dim, [3,]*dim) sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, dim, dtype=dtype, device=device)
    vx = voxel_size.square().reciprocal()

    # build sparse kernel
    kernel = []
    center_index = [1] * dim
    indices = []
    for d in range(dim):  # input channel
        kernel += [2 + 2 * vx.sum() / vx[d]]
        index = [d, d, *center_index]
        indices.append(index)
        for dd in range(dim):  # cross
            if dd == d:
                kernel += [-2] * 2
            else:
                kernel += [-vx[dd] / vx[d]] * 2
            index = [d, d, *center_index]
            index[2 + dd] = 0
            indices.append(index)
            index = [d, d, *center_index]
            index[2 + dd] = 2
            indices.append(index)
        for dd in range(d + 1, dim):  # output channel
            kernel += [-0.25] * 4
            index = [d, dd, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 0
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 0
            indices.append(index)
            index = [d, dd, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 2
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 2
            indices.append(index)
            kernel += [0.25] * 4
            index = [d, dd, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 2
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 0
            index[2 + dd] = 2
            indices.append(index)
            index = [d, dd, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 0
            indices.append(index)
            index = [dd, d, *center_index]
            index[2 + d] = 2
            index[2 + dd] = 0
            indices.append(index)

    indices = torch.as_tensor(indices, dtype=torch.long, device=vx.device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel,
                                     [dim, dim] + [3] * dim)

    return kernel


def lame_div_kernel(dim, voxel_size=1, dtype=None, device=None):
    """Precision matrix for the Linear Elastic energy, as a convolution kernel.

    Note
    ----
    .. Specialized implementation for the l2 version (i.e., no weight map).
    .. This is exactly equivalent to SPM's LE energy
    .. The convolution can be performed with `ni.spatial.spconv`

    Parameters
    ----------
    dim : int
    voxel_size : float or sequence[float], default=1 (actually unused)

    Returns
    -------
    kernel : (dim, dim, [3,]*dim) sparse tensor

    """
    dtype = dtype or torch.get_default_dtype()

    # build sparse kernel
    kernel = []
    center_index = [1] * dim
    indices = []
    for d in range(dim):  # input channel
        kernel += [2]
        index = [d, d, *center_index]
        indices.append(index)
        kernel += [-1] * 2
        index = [d, d, *center_index]
        index[2 + d] = 0
        indices.append(index)
        index = [d, d, *center_index]
        index[2 + d] = 2
        indices.append(index)
        for dd in range(d + 1, dim):  # output channel
            for d1 in range(dim):  # interation 1
                for d2 in range(d + 1, dim):  # interation 2
                    kernel += [-0.25] * 4
                    index = [d, dd, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 0
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 0
                    indices.append(index)
                    index = [d, dd, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 2
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 2
                    indices.append(index)
                    kernel += [0.25] * 4
                    index = [d, dd, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 2
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 0
                    index[2 + d2] = 2
                    indices.append(index)
                    index = [d, dd, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 0
                    indices.append(index)
                    index = [dd, d, *center_index]
                    index[2 + d1] = 2
                    index[2 + d2] = 0
                    indices.append(index)

    indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    kernel = torch.sparse_coo_tensor(indices.t(), kernel,
                                     [dim, dim] + [3] * dim)

    return kernel


def sum_kernels(dim, *kernels):
    """Sum sparse kernels of different shapes

    Parameters
    ----------
    dim : int
        Number of spatial dimensions
    *kernels : sparse tensor
        Kernels to sum

    Returns
    -------
    kernel : sparse tensor
        Sum of input tensors

    """
    # compute maximum shape
    spatial = [0] * dim
    device = None
    dtype = None
    for k in kernels:
        device = k.device
        dtype = k.dtype
        kspatial = k.shape[-dim:]
        spatial = [max(s, ks) for s, ks in zip(spatial, kspatial)]
    has_matrix = any(kernel.dim() == dim + 2 for kernel in kernels)
    has_diag = any(kernel.dim() == dim + 1 for kernel in kernels)

    # prepare output
    out_shape = [dim, dim] if has_matrix else [dim] if has_diag else []
    out_shape += spatial
    out = torch.sparse_coo_tensor(
        torch.zeros([len(out_shape), 0], dtype=torch.long, device=device),
        torch.zeros([0], dtype=dtype, device=device),
        out_shape)

    # sum kernels
    for kernel in kernels:
        offset = [(s - ks) // 2 for s, ks in zip(spatial, kernel.shape[-dim:])]
        if any(offset):
            new_shape = [*kernel.shape[:-dim], *spatial]
            offset = torch.as_tensor(offset,
                                     **get_backend(kernel._indices()))
            indices = kernel._indices()
            indices[-dim:] += offset[:, None]
            kernel = torch.sparse_coo_tensor(
                indices, kernel._values(), new_shape)
        if has_matrix:
            if kernel.dim() == dim:
                for d in range(len(out)):
                    pad_indices = torch.full([], d, dtype=torch.long,
                                             device=kernel.device)
                    pad_indices = pad_indices.expand(
                        [2, kernel._indices().shape[-1]])
                    indices = torch.cat([pad_indices, kernel._indices()], 0)
                    new_kernel = torch.sparse_coo_tensor(
                        indices, kernel._values(), out_shape)
                    out += new_kernel
            elif kernel.dim() == dim + 1:
                for d in range(len(out)):
                    pad_indices = torch.full([], d, dtype=torch.long,
                                             device=kernel.device)
                    pad_indices = pad_indices.expand(
                        [2, kernel[d]._indices().shape[-1]])
                    indices = torch.cat([pad_indices, kernel[d]._indices()], 0)
                    new_kernel = torch.sparse_coo_tensor(
                        indices, kernel[d]._values(), out_shape)
                    out += new_kernel
            else:
                out += kernel
        else:
            out += kernel

    out = out.coalesce()
    return out


def regulariser_grid(v, absolute=0, membrane=0, bending=0, lame=0,
                     factor=1, voxel_size=1, bound='dft', weights=None):
    """Precision matrix for a mixture of energies for a deformation grid.

    Parameters
    ----------
    v : (..., *spatial, dim) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : [sequence of] float, default=1
    bound : str, default='dft'
    weights : [dict of] (..., *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending', 'lame'}
        Else: the same weight map is shared across penalties.

    Returns
    -------
    Lv : (..., *spatial, dim) tensor

    """
    backend = dict(dtype=v.dtype, device=v.device)
    dim = v.shape[-1]

    if not (absolute or membrane or bending or any(ensure_list(lame))):
        return torch.zeros_like(v)

    voxel_size = make_vector(voxel_size, dim, **backend)
    absolute = absolute * factor
    membrane = membrane * factor
    bending = bending * factor
    lame = make_vector(lame, 2).tolist()
    lame = [l * factor for l in lame]
    fdopt = dict(bound=bound, voxel_size=voxel_size)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
        wl = weights.get('lame', None)
    else:
        wa = wm = wb = wl = weights
    wl = ensure_list(wl, 2)

    y = torch.zeros_like(v)
    if absolute:
        y.add_(absolute_grid(v, weights=wa, voxel_size=voxel_size),
               alpha=absolute)
    if membrane:
        y.add_(membrane_grid(v, weights=wm, **fdopt), alpha=membrane)
    if bending:
        y.add_(bending_grid(v, weights=wb, **fdopt), alpha=bending)
    if lame[0]:
        y.add_(lame_shear(v, weights=wl[0], **fdopt), alpha=lame[1])
    if lame[1]:
        y.add_(lame_div(v, weights=wl[1], **fdopt), alpha=lame[0])

    return y


def regulariser_grid_kernel(dim, absolute=0, membrane=0, bending=0, lame=0,
                            factor=1, voxel_size=1, dtype=None, device=None):
    """Precision kernel for a mixture of energies for a deformation grid.

    Parameters
    ----------
    dim : int
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    lame : (float, float), default=0
    factor : float, default=1
    voxel_size : [sequence of] float, default=1

    Returns
    -------
    kernel : ([dim, [dim,]] *spatial) sparse tensor

    """
    lame_shear, lame_div = ensure_list(lame, 2)
    backend = dict(dtype=dtype, device=device)

    kernels = []
    if absolute:
        kernels.append(
            absolute_grid_kernel(dim, voxel_size, **backend) * absolute)
    if membrane:
        kernels.append(
            membrane_grid_kernel(dim, voxel_size, **backend) * membrane)
    if bending:
        kernels.append(
            bending_grid_kernel(dim, voxel_size, **backend) * bending)
    if lame_shear:
        kernels.append(
            lame_shear_kernel(dim, voxel_size, **backend) * lame_shear)
    if lame_div:
        kernels.append(lame_div_kernel(dim, voxel_size, **backend) * lame_div)
    kernel = sum_kernels(dim, *kernels)
    kernel *= factor
    return kernel


def regulariser(x, absolute=0, membrane=0, bending=0, factor=1,
                voxel_size=1, bound='dct2', dim=None, weights=None):
    """Precision matrix for a mixture of energies.

    Parameters
    ----------
    x : (..., K, *spatial) tensor
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    factor : (sequence of) float, default=1
    voxel_size : (sequence of) float, default=1
    bound : str, default='dct2'
    dim : int, default=`gradient.dim()-1`
    weights : [dict of] (..., 1|K, *spatial) tensor, optional
        If a dict: keys must be in {'absolute', 'membrane', 'bending'}
        Else: the same weight map is shared across penalties.

    Returns
    -------
    Lx : (..., K, *spatial) tensor

    """
    if not (absolute or membrane or bending):
        return torch.zeros_like(x)

    dim = dim or (x.dim() - 1)
    backend = dict(dtype=x.dtype, device=x.device)
    dim = dim or x.dim() - 1
    nb_prm = x.shape[-dim - 1]

    voxel_size = make_vector(voxel_size, dim, **backend)
    factor = make_vector(factor, nb_prm, **backend)
    fdopt = dict(bound=bound, voxel_size=voxel_size, dim=dim)
    if isinstance(weights, dict):
        wa = weights.get('absolute', None)
        wm = weights.get('membrane', None)
        wb = weights.get('bending', None)
    else:
        wa = wm = wb = weights

    absolute = ensure_list(absolute, nb_prm) if absolute else []
    membrane = ensure_list(membrane, nb_prm) if membrane else []
    bending = ensure_list(bending, nb_prm) if bending else []

    wa = wa.expand(x.shape) if wa is not None else [None] * nb_prm
    wm = wm.expand(x.shape) if wm is not None else [None] * nb_prm
    wb = wb.expand(x.shape) if wb is not None else [None] * nb_prm

    y = torch.zeros_like(x)
    for x1, y1, w1, alpha in zip(x, y, wa, absolute):
        if alpha:
            y1.add_(_absolute(x1, weights=w1), alpha=alpha)
    for x1, y1, w1, alpha in zip(x, y, wm, membrane):
        if alpha:
            y1.add_(_membrane(x1, weights=w1, **fdopt), alpha=alpha)
    for x1, y1, w1, alpha in zip(x, y, wb, bending):
        if alpha:
            y1.add_(_bending(x1, weights=w1, **fdopt), alpha=alpha)

    pad_spatial = (Ellipsis,) + (None,) * dim
    return y.mul_(factor[pad_spatial])


def regulariser_kernel(dim, absolute=0, membrane=0, bending=0,
                       factor=1, voxel_size=1, dtype=None, device=None):
    """Precision kernel for a mixture of energies for a deformation grid.

    Parameters
    ----------
    dim : int
    absolute : float, default=0
    membrane : float, default=0
    bending : float, default=0
    factor : float, default=1
    voxel_size : [sequence of] float, default=1

    Returns
    -------
    kernel : ([dim, [dim,]] *spatial) sparse tensor

    """
    backend = dict(dtype=dtype, device=device)

    kernels = []
    if absolute:
        kernels.append(absolute_kernel(dim, voxel_size, **backend))
    if membrane:
        kernels.append(membrane_kernel(dim, voxel_size, **backend))
    if bending:
        kernels.append(bending_kernel(dim, voxel_size, **backend))
    kernel = sum_kernels(dim, *kernels)
    kernel *= factor
    return kernel
