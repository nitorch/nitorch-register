import torch
from nitorch_interpol import (
    resize_flow, add_identity_grid, add_identity_grid_
)

# ----------------------------------------------------------------------
#                           HELPERS
# ----------------------------------------------------------------------


def samespace(aff, inshape, outshape):
    """Check if two spaces are the same

    Parameters
    ----------
    aff : (dim+1, dim+1)
    inshape : tuple of `dim` int
    outshape : tuple of `dim` int

    Returns
    -------
    bool

    """
    dim = len(inshape)
    eye = torch.eye(dim+1, dtype=aff.dtype, device=aff.device)
    return inshape == outshape and aff.allclose(eye)


def ffd_exp(prm, shape, order=3, bound='dft', returns='disp'):
    """Transform FFD parameters into a displacement or transformation grid.

    Parameters
    ----------
    prm : (..., *spatial, dim)
        FFD parameters
    shape : sequence[int]
        Exponentiated shape
    order : int, default=3
        Spline order
    bound : str, default='dft'
        Boundary condition
    returns : {'disp', 'grid', 'disp+grid'}, default='grid'
        What to return:
        - 'disp' -> displacement grid
        - 'grid' -> transformation grid

    Returns
    -------
    disp : (..., *shape, dim), optional
        Displacement grid
    grid : (..., *shape, dim), optional
        Transformation grid

    """
    dim = prm.shape[-1]
    batch = prm.shape[:-(dim + 1)]
    prm = prm.reshape([-1, *prm.shape[-(dim + 1):]])
    disp = resize_flow(prm, shape=shape, order=order, bound=bound)
    disp = disp.reshape(batch + disp.shape[1:])
    if 'disp' in returns and 'grid' in returns:
        grid = add_identity_grid(disp)
        return disp, grid
    elif 'disp' in returns:
        return disp
    elif 'grid' in returns:
        return add_identity_grid_(disp)


class BacktrackingLineSearch(torch.optim.Optimizer):

    def __init__(self, optim, armijo=1, max_iter=6):
        self.optim = optim
        self.armijo = float(armijo)
        self.max_iter = max_iter

    def step(self, closure, loss=None):
        """

        Parameters
        ----------
        closure : callable
        loss : tensor, optional

        Returns
        -------
        tensor

        """

        def get_params():
            params = []
            for group in self.optim.param_groups:
                params.extend(group['params'])
            return params

        with torch.no_grad():
            if loss is None:
                loss = closure()
            armijo = self.armijo

            params0 = [param.detach().clone() for param in get_params()]
            self.optim.step()
            deltas = [p - p0 for p, p0 in zip(get_params(), params0)]
            self.last_ok = False
            for n_iter in range(self.max_iter):
                new_loss = closure()
                if new_loss < loss:
                    armijo = 2 * armijo
                    self.last_ok = True
                    break
                else:
                    armijo = armijo / 2.
                    for param, param0, delta in zip(get_params(), params0,
                                                    deltas):
                        param.copy_(param0 + armijo * delta)
        return new_loss

    def add_param_group(self, param_group: dict) -> None:
        return self.optim.add_param_group(param_group)

    def load_state_dict(self, state_dict: dict) -> None:
        return self.optim.load_state_dict(state_dict)

    def state_dict(self):
        return self.optim.state_dict()

    def zero_grad(self, *args, **kwargs) -> None:
        return self.optim.zero_grad(*args, **kwargs)

    @property
    def param_groups(self):
        return self.optim.param_groups
