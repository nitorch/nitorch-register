"""Integrate stationary velocity fields."""

import torch
from nitorch_core.optionals import custom_fwd, custom_bwd
from nitorch_fastmath import matvec, sym_matmul
from nitorch_interpol import (
    pull, identity_grid, flow_jacobian, identity_grid_like, add_identity_grid
)


__all__ = ['exp', 'exp_forward', 'exp_backward']


def exp(vel, inverse=False, steps=8, interpolation='linear', bound='dft',
        displacement=False, anagrad=False):
    """Exponentiate a stationary velocity field by scaling and squaring.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Stationary velocity field.
    inverse : bool, default=False
        Generate the inverse transformation instead of the forward.
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    interpolation : {0..7}, default=1
        Interpolation order
    bound : str, default='dft'
        Boundary conditions
    displacement : bool, default=False
        Return a displacement field rather than a transformation field
    anagrad : bool, default=False
        Use analytical gradients rather than autodiff gradients in
        the backward pass. Should be more memory efficient and (maybe)
        faster.

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Exponentiated tranformation

    """
    exp_fn = _Exp.apply if anagrad else exp_forward
    return exp_fn(vel, inverse, steps, interpolation, bound, displacement)


def exp_forward(vel, inverse=False, steps=8, interpolation='linear',
                bound='dft', displacement=False, jacobian=False,
                _anagrad=False):
    """Exponentiate a stationary velocity field by scaling and squaring.

    This function always uses autodiff in the backward pass.
    It can also compute Jacobian fields on the fly.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Stationary velocity field.
    inverse : bool, default=False
        Generate the inverse transformation instead of the forward.
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    interpolation : {0..7}, default=1
        Interpolation order
    bound : str, default='dft'
        Boundary conditions
    displacement : bool, default=False
        Return a displacement field rather than a transformation field

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Exponentiated tranformation

    """
    backend = dict(dtype=vel.dtype, device=vel.device)
    vel = -vel if inverse else vel.clone()

    # Precompute identity + aliases
    dim = vel.shape[-1]
    spatial = vel.shape[-1-dim:-1]
    id = identity_grid(spatial, **backend)
    jac = torch.eye(dim, **backend).expand([*vel.shape[:-1], dim, dim])
    opt = {'order': interpolation, 'bound': bound}

    if not _anagrad and vel.requires_grad:
        iadd = lambda x, y: x.add(y)
    else:
        iadd = lambda x, y: x.add_(y)

    vel /= (2**steps)
    for i in range(steps):
        if jacobian:
            jac = _composition_jac(jac, vel)
        vel = iadd(vel, pull(vel, id + vel, **opt))

    if not displacement:
        vel += id
    return (vel, jac) if jacobian else vel


def exp_backward(vel, *gradhess, inverse=False, steps=8,
                 interpolation='linear', bound='dft', rotate_grad=False):
    """Backward pass of SVF exponentiation.

    This should be much more memory-efficient than the autograd pass
    as we don't have to store intermediate grids.

    I am using DARTEL's derivatives (from the code, not the paper).
    From what I get, it corresponds to pushing forward the gradient
    (computed in observation space) recursively while squaring the
    (inverse) transform.
    Remember that the push forward of g by phi is
                    |iphi| iphi' * g(iphi)
    where iphi is the inverse of phi. We could also have implemented
    this operation as: inverse(phi)' * push(g, phi), since
    push(g, phi) \approx |iphi| g(iphi). It has the advantage of using
    push rather than pull, which might preserve better positive-definiteness
    of the Hessian, but requires the inversion of (potentially ill-behaved)
    Jacobian matrices.

    Note that gradients must first be rotated using the Jacobian of
    the exponentiated transform so that the denominator refers to the
    initial velocity (we want dL/dV0, not dL/dPsi).
    THIS IS NOT DONE INSIDE THIS FUNCTION YET (see _dartel).

    Parameters
    ----------
    vel : (..., *spatial, dim) tensor
        Velocity
    grad : (..., *spatial, dim) tensor
        Gradient with respect to the output grid
    hess : (..., *spatial, dim*(dim+1)//2) tensor, optional
        Symmetric hessian with respect to the output grid.
    inverse : bool, default=False
        Whether the grid is an inverse
    steps : int, default=8
        Number of scaling and squaring steps
    interpolation : str or int, default='linear'
    bound : str, default='dft'
    rotate_grad : bool, default=False
        If True, rotate the gradients using the Jacobian of exp(vel).

    Returns
    -------
    grad : (..., *spatial, dim) tensor
        Gradient with respect to the SVF
    hess : (..., *spatial, dim*(dim+1)//2) tensor, optional
        Approximate (block diagonal) Hessian with respect to the SVF

    """
    grad, *gradhess = gradhess
    has_hess = len(gradhess) > 0
    hess = gradhess[0] if has_hess else None

    opt = dict(bound=bound, order=interpolation)
    dim = vel.shape[-1]
    id = identity_grid_like(vel)
    vel = vel.clone()

    if rotate_grad:
        # It forces us to perform a forward exponentiation, which
        # is a bit annoying...
        # Maybe save the Jacobian after the forward pass? But it take space
        _, jac = exp_forward(vel, jacobian=True, steps=steps,
                             displacement=True, **opt, _anagrad=True)
        jac = jac.transpose(-1, -2)
        grad = matvec(jac, grad)
        if hess is not None:
            hess = sym_matmul(jac, hess)
        del jac

    vel /= (-1 if not inverse else 1) * (2**steps)
    jac = flow_jacobian(vel, bound=bound)

    # rotate gradient a bit so that when steps == 0, we get the same
    # gradients as the smalldef case
    ijac = 2 * torch.eye(dim, dtype=jac.dtype, device=jac.device) - jac
    ijac = ijac.transpose(-1, -2).inverse()
    grad = matvec(ijac, grad)
    del ijac

    for _ in range(steps):
        det = jac.det()
        jac = jac.transpose(-1, -2)
        grad0 = grad
        grad = pull(grad, id + vel, **opt)  # \
        grad = matvec(jac, grad)            # | push forward
        grad *= det[..., None]              # /
        grad += grad0                       # add all scales (SVF)
        if hess is not None:
            hess0 = hess
            hess = pull(hess, id + vel, **opt)
            hess = sym_matmul(jac, hess)
            hess *= det[..., None]
            hess += hess0
        # squaring
        jac = jac.transpose(-1, -2)
        jac = _composition_jac(jac, vel, **opt)
        vel += pull(vel, id + vel, **opt)

    if inverse:
        grad.neg_()

    grad /= (2**steps)
    if hess is not None:
        hess /= (2**steps)

    return (grad, hess) if has_hess else grad


class _Exp(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # det() only implemented in f32
    def forward(ctx, vel, inverse, steps, interpolation, bound, displacement):
        if vel.requires_grad:
            ctx.save_for_backward(vel)
            ctx.args = {'steps': steps, 'inverse': inverse,
                        'order': interpolation, 'bound': bound}
        return exp_forward(vel, inverse, steps, interpolation, bound,
                           displacement, _anagrad=True)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        vel, = ctx.saved_tensors
        grad = exp_backward(vel, grad,
                            steps=ctx.args['steps'],
                            inverse=ctx.args['inverse'],
                            interpolation=ctx.args['order'],
                            bound=ctx.args['bound'],
                            rotate_grad=True)
        return (grad,) + (None,)*5


def _composition_jac(jac, rhs, **kwargs):
    """Jacobian of the composition `(lhs)o(rhs)`

    Parameters
    ----------
    jac : ([batch], *spatial, ndim, ndim) tensor
        Jacobian of input RHS transformation
    rhs : ([batch], *spatial, ndim) tensor
        RHS transformation
    lhs : ([batch], *spatial, ndim) tensor, default=`rhs`
        LHS small displacement
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    composed_jac : ([batch], *spatial, ndim, ndim) tensor
        Jacobian of composition

    """
    ndim = rhs.shape[-1]
    jac_left = flow_jacobian(rhs)
    jac_left = jac_left.reshape(jac_left.shape[:-2] + (-1,))
    jac_left = pull(jac_left, add_identity_grid(rhs), **kwargs)
    jac_left = jac_left.reshape(jac_left.shape[:-1] + (ndim, ndim))
    jac = torch.matmul(jac_left, jac)
    return jac


