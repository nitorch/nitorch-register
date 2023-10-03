import nitorch_interpol as interpol
from jitfields.pushpull_backward import grad_backward
from nitorch_core import py
from nitorch_core.conv import smooth
import torch


class JointHist:
    """
    Joint histogram with a backward pass for optimization-based registration.
    """

    def __init__(self, n=64, order=3, fwhm=2, bound='replicate', extrapolate=False):
        """

        Parameters
        ----------
        n : int, default=64
            Number of bins
        order : int, default=3
            B-spline order
        bound : {'zero', 'replicate'}
            How to deal with out-of-bound values
        extrapolate : bool, default=False

        """
        self.n = py.make_list(n, 2)
        self.order = order
        self.bound = bound
        self.extrapolate = extrapolate
        self.fwhm = fwhm

    def _prepare(self, x, min, max):
        """

        Parameters
        ----------
        x : (..., N, 2) tensor
        min : (..., 2) tensor
        max : (..., 2) tensor

        Returns
        -------
        x : (batch, N, 2) tensor
            Reshaped and index-converted input

        """
        if min is None:
            min = x.detach().min(dim=-2).values
        min = min.unsqueeze(-2)
        if max is None:
            max = x.detach().max(dim=-2).values
        max = max.unsqueeze(-2)

        # apply affine function to transform intensities into indices
        x = x.clone()
        nn = torch.as_tensor(self.n, dtype=x.dtype, device=x.device)
        x = x.mul_(nn / (max - min)).add_(nn / (1 - max / min)).sub_(0.5)

        # reshape as (B, N, 2)
        x = x.reshape([-1, *x.shape[-2:]])
        return x, min.squeeze(-2), max.squeeze(-2)

    def forward(self, x, min=None, max=None, mask=None):
        """

        Parameters
        ----------
        x : (..., N, 2) tensor
            Input multivariate vector
        min : (..., 2) tensor, optional
        max : (..., 2) tensor, optional
        mask : (..., N) tensor, optional

        Returns
        -------
        h : (..., B, B) tensor
            Joint histogram

        """
        shape = x.shape
        x, min, max = self._prepare(x, min, max)  # x = [batch, vox, 2]

        # push data into the histogram
        #   hidden feature: tell pullpush to use +/- 0.5 tolerance when
        #   deciding if a coordinate is inbounds.
        extrapolate = self.extrapolate or 'edge'
        if mask is None:
            h = interpol.count(
                x[:, None],         # [batch, 1, vox, 2]
                self.n,             # (bin_mov, bin_fix)
                self.order,
                self.bound,
                extrapolate,
            )                       # -> [batch, bin_mov, bin_fix]
        else:
            mask = mask.to(x.device, x.dtype)
            h = interpol.push(
                mask[..., None],    # [(batch), 1, vox, 1]
                x[:, None],         # [batch, 1, vox, 2]
                self.n,             # (bin_mov, bin_fix)
                self.order,
                self.bound,
                extrapolate,
            )                       # [batch, bin_mov, bin_fix, 1]
            h = h.squeeze(-1)       # [batch, bin_mov, bin_fix]
        h = h.to(x.dtype)
        h = h.reshape([*shape[:-2], *h.shape[-2:]])

        if self.fwhm:
            h = smooth(h, fwhm=self.fwhm, bound=self.bound, dim=2)

        return h, min, max

    def backward(self, x, g, min=None, max=None, hess=False, mask=None):
        """

        Parameters
        ----------
        x : (..., N, 2) tensor
            Input multidimensional vector
        g : (..., B, B) tensor
            Gradient with respect to the histogram
        min : (..., 2) tensor, optional
        max : (..., 2) tensor, optional

        Returns
        -------
        g : (..., N, 2) tensor
            Gradient with respect to x

        """
        if self.fwhm:
            g = smooth(g, fwhm=self.fwhm, bound=self.bound, dim=2)

        shape = x.shape
        x, min, max = self._prepare(x, min, max)  # x = [batch, vox, 2]
        nvox = x.shape[-2]
        min = min.unsqueeze(-2)
        max = max.unsqueeze(-2)
        g = g.reshape([-1, *g.shape[-2:]])        # g = [batch, bin_mov bin_fix]

        extrapolate = self.extrapolate or 2
        if not hess:
            g = interpol.grad(
                g[:, :, :, None],        # [batch, bin_mov, bin_fix, 1]
                x[:, None, :, :],        # [batch, 1,       vox,     2]
                self.order,
                self.bound,
                extrapolate,
            )                           # [batch, 1, vox, 1, 2]
            g = g.reshape(shape)        # [*batch, *spatial, 2]
        else:
            # 1) Absolute value of adjoint of gradient
            o = torch.ones_like(x)
            o, _ = grad_backward(
                o[:, None, :, None, :],  # [batch, 1,       vox,     1, 2]
                g[:, :, :, None],        # [batch, bin_mov, bin_fix, 1]
                x[:, None, :, :],        # [batch, 1,       vox,     2]
                self.bound,
                self.order,
                extrapolate,
            )                            # [batch, bin_mov, bin_fix, 1]
            g *= o.squeeze(-1)           # [batch, bin_mov, bin_fix]
            # 2) Absolute value of gradient
            #   g : [batch=1, channel=1, spatial=[B(mov), B(fix)]]
            #   x : [batch=1, spatial=[1, vox], dim=2]
            #    -> [batch=1, channel=1, spatial=[1, vox], 2]
            g = interpol.grad(
                g[:, :, :, None],        # [batch, bin_mov, bin_fix, 1]
                x[:, None, :, :],        # [batch, 1,       vox,     2]
                self.bound,
                self.order,
                extrapolate,
            )                            # [batch, 1,       vox, 1, 2]
            g = g.reshape(shape)         # [*batch, *spatial, 2]

        # adjoint of affine function
        nn = torch.as_tensor(self.n, dtype=x.dtype, device=x.device)
        factor = nn / (max - min)
        if hess:
            factor = factor.square_()
        g = g.mul_(factor)
        if mask is not None:
            g *= mask[..., None]

        return g


class TestJointHist(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return JointHist().forward(x)[0]

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return JointHist().backward(x, g)

