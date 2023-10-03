from nitorch_interpol import (
    push, pushcount, add_identity_grid_, add_identity_grid, sub_identity_grid
)
from nitorch_solvers.flows import flow_solve_fmg


def flow_inv(grid, type='grid', bound='dft', extrapolate=True, **prm):
    r"""Invert a dense deformation (or displacement) grid

    Notes
    -----
    The deformation/displacement grid must be expressed in
    voxels, and map from/to the same lattice.

    Let `f = id + d` be the transformation. The inverse
    is obtained as `id - (f.T @ 1 + L) \ (f.T @ d)`
    where `L` is a regulariser, `f.T @ _` is the adjoint
    operation ("push") of `f @ _` ("pull"). and `1` is an
    image of ones.

    The idea behind this is that `f.T @ _` is approximately
    the inverse transformation weighted by the determinant
    of the Jacobian of the tranformation so, in the (theoretical)
    continuous case, `inv(f) @ _ = f.T @ _ / f.T @ 1`.
    However, in the (real) discrete case, this leads to
    lots of holes in the inverse. The solution we use
    therefore fills these holes using a regularised
    least-squares scheme, where the regulariser penalizes
    the spatial gradients of the inverse field.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation (or displacement) grid
    type : {'grid', 'disp'}, default='grid'
        Type of deformation.
    membrane : float, default=0.1
        Regularisation
    bound : str, default='dft'
        Boundary conditions
    extrapolate : bool, default=True
        Extrapolate the transformation field when
        it is sampled out-of-bounds.

    Returns
    -------
    grid_inv : (..., *spatial, dim)
        Inverse transformation (or displacement) grid

    """
    prm = prm or dict(membrane=0.1)

    # get displacement
    if type == 'grid':
        disp = sub_identity_grid(grid)
    else:
        disp = grid
        grid = add_identity_grid(disp)

    # push displacement
    push_opt = dict(bound=bound, extrapolate=extrapolate)
    disp = push(disp, grid, **push_opt)
    count = pushcount(grid, **push_opt).unsqueeze(-1)

    # Fill missing values using regularised least squares
    disp = flow_solve_fmg(count, disp, bound=bound, **prm)

    disp = disp.neg_()
    if type == 'grid':
        disp = add_identity_grid_(disp)
    return disp