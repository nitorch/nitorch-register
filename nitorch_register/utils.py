"""Utility functions for registration algorithms.
Some (jg, jhj, affine_grid_backward) should maybe be moved to the `spatial`
module (?).
"""

import torch
from nitorch_interpol import (
    pull, push, sub_identity_grid, identity_grid_like,
)
from nitorch_core.py import prod
from . import svf


def defaults_velocity(prm=None):
    if prm is None:
        prm = dict()
    # values from SPM shoot
    prm.setdefault('absolute', 1e-4)
    prm.setdefault('membrane', 1e-3)
    prm.setdefault('bending', 0.2)
    prm.setdefault('lame', (0.05, 0.2))
    prm.setdefault('voxel_size', 1.)
    return prm


def defaults_template(prm=None):
    if prm is None:
        prm = dict()
    # values from SPM shoot
    prm.setdefault('absolute', 1e-4)
    prm.setdefault('membrane', 0.08)
    prm.setdefault('bending', 0.8)
    prm.setdefault('voxel_size', 1.)
    return prm


def loadf(x):
    """Load data from disk if needed"""
    return x.fdata() if hasattr(x, 'fdata') else x


def savef(x, parent):
    """Save data to disk if needed"""
    if hasattr(parent, 'fdata'):
        parent[...] = x
    else:
        parent.copy_(x)


def smart_pull(image, grid, **kwargs):
    """pull that accepts None grid"""
    if image is None or grid is None:
        return image
    return pull(image, grid, **kwargs)


def smart_push(image, grid, **kwargs):
    """push that accepts None grid"""
    if image is None or grid is None:
        return image
    return push(image, grid, **kwargs)


def smart_exp(vel, **kwargs):
    """exp that accepts None vel"""
    if vel is not None:
        vel = svf.exp(vel, **kwargs)
    return vel


def smart_pull_grid(vel, grid, *args, **kwargs):
    """Interpolate a velocity/grid/displacement field.

    Notes
    -----
    Defaults differ from grid_pull:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    vel : ([batch], *spatial, ndim) tensor
        Velocity
    grid : ([batch], *spatial, ndim) tensor
        Transformation field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : ([batch], *spatial, ndim) tensor
        Velocity

    """
    if grid is None or vel is None:
        return vel
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    if type == 'grid':
        vel = sub_identity_grid(vel)
    vel = pull(vel, grid, *args, **kwargs)
    return vel


def smart_pull_jac(jac, grid, *args, **kwargs):
    """Interpolate a jacobian field.

    Notes
    -----
    Defaults differ from grid_pull:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    jac : ([batch], *spatial_in, ndim, ndim) tensor
        Jacobian field
    grid : ([batch], *spatial_out, ndim) tensor
        Transformation field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_jac : ([batch], *spatial_out, ndim) tensor
        Jacobian field

    """
    if grid is None or jac is None:
        return jac
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    dim = jac.shape[-1]
    jac = jac.reshape([*jac.shape[:-2], dim*dim])  # collapse matrix
    jac = pull(jac, grid, *args, **kwargs)
    jac = jac.reshape([*jac.shape[:-1], dim, dim])
    return jac


def smart_push_grid(vel, grid, *args, **kwargs):
    """Push a velocity/grid/displacement field.

    Notes
    -----
    Defaults differ from grid_push:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    vel : ([batch], *spatial, ndim) tensor
        Velocity
    grid : ([batch], *spatial, ndim) tensor
        Transformation field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : ([batch], *spatial, ndim) tensor
        Velocity

    """
    if grid is None or vel is None:
        return vel
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    vel = push(vel, grid, *args, **kwargs)
    return vel


@torch.jit.script
def _affine_grid_backward_g(grid, grad):
    dim = grid.shape[-1]
    g = torch.empty([grad.shape[0], dim, dim+1], dtype=grad.dtype, device=grad.device)
    for i in range(dim):
        g[:, i, -1] = grad[:, :, i].sum(1, dtype=torch.double).to(g.dtype)
        for j in range(dim):
            g[:, i, j] = (grad[:, :, i] * grid[:, :, j]).sum(1, dtype=torch.double).to(g.dtype)
    return g


@torch.jit.script
def _affine_grid_backward_gh(grid, grad, hess):
    dim = grid.shape[-1]
    g = torch.zeros([grad.shape[0], dim, dim+1], dtype=grad.dtype, device=grad.device)
    h = torch.zeros([hess.shape[0], dim, dim+1, dim, dim+1], dtype=grad.dtype, device=grad.device)
    basecount = dim - 1
    for i in range(dim):
        basecount = basecount + i * (dim-i)
        for j in range(dim+1):
            if j == dim:
                g[:, i, j] = (grad[:, :, i]).sum(1)
            else:
                g[:, i, j] = (grad[:, :, i] * grid[:, :, j]).sum(1)
            for k in range(dim):
                idx = k
                if k < i:
                    continue
                elif k != i:
                    idx = basecount + (k - i)
                for l in range(dim+1):
                    if l == dim and j == dim:
                        h[:, i, j, k, l] = h[:, k, j, i, l] = hess[:, :, idx].sum(1)
                    elif l == dim:
                        h[:, i, j, k, l] = h[:, k, j, i, l] = (hess[:, :, idx] * grid[:, :, j]).sum(1)
                    elif j == dim:
                        h[:, i, j, k, l] = h[:, k, j, i, l] = (hess[:, :, idx] * grid[:, :, l]).sum(1)
                    else:
                        h[:, i, j, k, l] = h[:, k, j, i, l] = (hess[:, :, idx] * grid[:, :, j] * grid[:, :, l]).sum(1)
    return g, h


def affine_grid_backward(*grad_hess, grid=None):
    """Converts ∇ wrt dense displacement into ∇ wrt affine matrix

    g = affine_grid_backward(g, [grid=None])
    g, h = affine_grid_backward(g, h, [grid=None])

    Parameters
    ----------
    grad : (..., *spatial, dim) tensor
        Gradient with respect to a dense displacement.
    hess : (..., *spatial, dim*(dim+1)//2) tensor, optional
        Hessian with respect to a dense displacement.
    grid : (*spatial, dim) tensor, optional
        Pre-computed identity grid

    Returns
    -------
    grad : (..., dim, dim+1) tensor
        Gradient with respect to an affine matrix
    hess : (..., dim, dim+1, dim, dim+1) tensor, optional
        Hessian with respect to an affine matrix

    """
    has_hess = len(grad_hess) > 1
    grad, *hess = grad_hess
    hess = hess.pop(0) if hess else None
    del grad_hess

    dim = grad.shape[-1]
    shape = grad.shape[-dim-1:-1]
    batch = grad.shape[:-dim-1]
    nvox = prod(shape)
    if grid is None:
        grid = identity_grid_like(grad)
    grid = grid.reshape([1, nvox, dim])
    grad = grad.reshape([-1, nvox, dim])
    if hess is not None:
        hess = hess.reshape([-1, nvox, dim*(dim+1)//2])
        grad, hess = _affine_grid_backward_gh(grid, grad, hess)
        hess = hess.reshape([*batch, dim, dim+1, dim, dim+1])
    else:
        grad = _affine_grid_backward_g(grid, grad)
    grad = grad.reshape([*batch, dim, dim+1])
    return (grad, hess) if has_hess else grad
