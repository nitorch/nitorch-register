import torch
from nitorch_fastmath import sym_solve, lmdiv
from nitorch_solvers.flows import flow_solve_fmg, flow_solve_cg, flow_solve_gs
from .base import SecondOrder


class SymGaussNewton(SecondOrder):
    """Base class for Gauss-Newton"""

    def __init__(self, lr=1, marquardt=True, preconditioner=None,
                 **kwargs):
        super().__init__(lr, **kwargs)
        self.preconditioner = preconditioner
        self.marquardt = marquardt

    def _add_marquardt(self, grad, hess, tiny=1e-5):
        dim = grad.shape[-1]
        if self.marquardt is True:
            # maj = hess[..., :dim].abs()
            # if hess.shape[-1] > dim:
            #     maj.add_(hess[..., dim:].abs(), alpha=2)
            maj = hess[..., :dim].abs().max(-1, True).values
            hess[..., :dim].add_(maj, alpha=tiny)
            # hess[..., :dim] += tiny
        elif self.marquardt:
            hess[..., :dim] += self.marquardt
        return grad, hess

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        step = sym_solve(hess, grad)
        step.mul_(-self.lr)
        return step


class GaussNewton(SecondOrder):
    """Base class for Gauss-Newton"""

    def __init__(self, lr=1, marquardt=True, preconditioner=None,
                 **kwargs):
        super().__init__(lr, **kwargs)
        self.preconditioner = preconditioner
        self.marquardt = marquardt

    def _add_marquardt(self, grad, hess, tiny=1e-5):
        dim = grad.shape[-1]
        if self.marquardt is True:
            # maj = hess[..., :dim].abs()
            # if hess.shape[-1] > dim:
            #     maj.add_(hess[..., dim:].abs(), alpha=2)
            maj = hess.diagonal(0, -1, -2).abs().max(-1, True).values
            hess.diagonal(0, -1, -2).add_(maj, alpha=tiny)
            # hess[..., :dim] += tiny
        elif self.marquardt:
            hess[..., :dim] += self.marquardt
        return grad, hess

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        step = lmdiv(hess, grad[..., None])[..., 0]
        step.mul_(-self.lr)
        return step


class FlowGaussNewton(SecondOrder):
    """Base class for Gauss-Newton on displacement grids"""

    def __init__(self, lr=1, fmg=2, max_iter=2, factor=1, voxel_size=1,
                 absolute=0, membrane=0, bending=0, lame=0, reduce='mean',
                 marquardt=True, preconditioner=None, **kwargs):
        super().__init__(lr, **kwargs)
        self.preconditioner = preconditioner
        self.factor = factor
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.lame = lame
        self.reduce = reduce in ('mean', True)
        self.voxel_size = voxel_size
        self.marquardt = marquardt
        self.max_iter = max_iter
        self.fmg = fmg

    @property
    def penalty(self):
        return dict(absolute=self.absolute,
                    membrane=self.membrane,
                    bending=self.bending,
                    lame=self.lame,
                    factor=self.factor)

    @penalty.setter
    def penalty(self, x):
        for key, value in x.items():
            if key in ('absolute', 'membrane', 'bending', 'lame', 'factor'):
                setattr(self, key, value)

    def _get_prm(self, nvox):
        factor = self.factor
        if self.reduce:
            factor = factor / nvox
        prm = dict(absolute=self.absolute * factor,
                   membrane=self.membrane * factor,
                   bending=self.bending * factor,
                   lame=self.lame * factor,
                   voxel_size=self.voxel_size)
        if self.fmg:
            prm['nb_iter'] = self.max_iter
            prm['nb_cycles'] = self.fmg
        else:
            prm['max_iter'] = self.max_iter
        return prm

    def _add_marquardt(self, grad, hess, tiny=1e-5):
        dim = grad.shape[-1]
        if self.marquardt is True:
            maj = hess[..., :dim].abs().max(-1, True).values
            hess[..., :dim].add_(maj, alpha=tiny)
        elif self.marquardt:
            hess[..., :dim] += self.marquardt
        return grad, hess

    def repr_keys(self):
        return super().repr_keys() + ['fmg']


class FlowCG(FlowGaussNewton):
    """Gauss-Newton on displacement grids using Conjugate Gradients"""

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        dim = grad.shape[-1]
        prm = self._get_prm(grad.shape[-dim-1:-1].numel())
        if self.fmg:
            step = flow_solve_fmg(hess, grad, solver='cg', **prm)
        else:
            step = flow_solve_cg(hess, grad, **prm)
        step.masked_fill_(torch.isfinite(step).bitwise_not_(), 0)
        step.mul_(-self.lr)
        return step


class FlowRelax(FlowGaussNewton):
    """Gauss-Newton on displacement grids using Relaxation"""

    def search_direction(self, grad, hess):
        grad, hess = self._add_marquardt(grad, hess)
        dim = grad.shape[-1]
        prm = self._get_prm(grad.shape[-dim-1:-1].numel())
        if self.fmg:
            step = flow_solve_fmg(hess, grad, solver='gs', **prm)
        else:
            step = flow_solve_gs(hess, grad, **prm)
        step.masked_fill_(torch.isfinite(step).bitwise_not_(), 0)
        step.mul_(-self.lr)
        return step
