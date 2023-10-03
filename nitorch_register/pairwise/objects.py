"""Object-oriented representation of images, losses, transforms, etc."""
import nitorch_io as io
import nitorch_fastmath as fastmath
import nitorch_interpol as interpol
from nitorch_core.py import make_list, prod
from nitorch_core.extra import backend as get_backend
from nitorch_core.finite_differences import diff
from nitorch_core.conv import pool, smooth
from nitorch_solvers.flows import flow_penalty
from nitorch_interpol import (
    resize_flow, resize_affine,
    pullgrad, flow_jacobian, affine_grid,
    identity_grid, add_identity_grid, sub_identity_grid,
)
from nitorch_register.affine import (
    affine_default, mean_space, voxel_size as get_voxel_size,
    affine_conv, affine_sub, affine_basis_size, affine_basis, affine_matvec
)
from nitorch_register.flow_inv import flow_inv
from nitorch_register.shoot import greens, greens_apply, shoot
from nitorch_register.svf import exp_forward
from nitorch_register import losses, utils as regutils
import torch
import copy


class BaseSimilarity:

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)


class Similarity(BaseSimilarity):
    """
    Similarity measure between two images

    A similarity term between a fixed and moving image is instantiated
    by e.g. `sim = Similarity(NCC(), mov, fix)`, where `mov` and `fix`
    are pre-instantiated `Image` objects.

    A Similarity instance can be multiplied or divided by a scalar number,
    with the effect of modulating its weight (one by default) in the loss.
    E.g., `loss = 10 * sim`.

    The `reverse()` method generates a function that expects `moving` to
    be warped to `fixed` using the inverse transform. The `symmetrize()`
    method generates a symmetric similarity that contains both the
    forward and reverse similarities.

    Multiple similarity terms can be combined by addition, e.g.
    `loss = 10 * sim1 + 0.5 * sim2`
    """

    def __init__(self, loss, moving, fixed, factor=1, backward=False):
        """

        Parameters
        ----------
        loss : OptimizationLoss (or its name)
            Similarity loss to minimize.
        moving, fixed : Image
            Fixed and moving images.
        factor : float, default=1
            Weight of the loss
        backward : bool, default=False
            If True, apply the backward transform instead of the forward.
        """
        self.moving = moving
        self.fixed = fixed
        self.loss = losses.make_loss(loss, self.moving.dim)
        self.factor = factor
        self.backward = backward

    def images(self):
        yield self.moving
        yield self.fixed

    def losses(self):
        yield self.loss

    def reverse(self):
        """
        Return the reverse similarity, where `fixed` is warped to `moving`
        """
        return type(self)(copy.deepcopy(self.loss),
                          self.fixed, self.moving, self.factor,
                          not self.backward)

    def symmetrize(self):
        """
        Return a symmetric loss, that includes both the forward and
        reverse losses.
        """
        return SumSimilarity([self, self.reverse()]) * 0.5

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __getitem__(self, item):
        if item > 0:
            raise IndexError
        return self

    def __mul__(self, factor: float):
        obj = self.copy()
        obj.factor = factor * obj.factor
        return obj

    def __rmul__(self, factor: float):
        obj = self.copy()
        obj.factor = obj.factor * factor
        return obj

    def __imul__(self, factor: float):
        self.factor *= factor
        return self

    def __truediv__(self, factor: float):
        obj = self.copy()
        obj.factor = factor / obj.factor
        return obj

    def __itruediv__(self, factor: float):
        self.factor /= factor
        return self

    def __add__(self, other: BaseSimilarity):
        if isinstance(other, (int, float)) and other == 0:
            return self
        if isinstance(other, SumSimilarity):
            return SumSimilarity([self, *other])
        elif isinstance(other, BaseSimilarity):
            return SumSimilarity([self, other])
        raise TypeError(f'Cannot add a {type(self)} and a {type(other)}')

    def __radd__(self, other: BaseSimilarity):
        if isinstance(other, (int, float)) and other == 0:
            return self
        if isinstance(other, SumSimilarity):
            return SumSimilarity([*other, self])
        elif isinstance(other, BaseSimilarity):
            return SumSimilarity([other, self])
        raise TypeError(f'Cannot add a {type(other)} and a {type(self)}')

    def _prm_as_str(self):
        s = []
        if self.backward:
            s += ['backward=True']
        p = getattr(self.loss, 'patch', None)
        if p is not None:
            s += [f'patch={p}']
        return s

    def __repr__(self):
        s = ', '.join(self._prm_as_str())
        s = f'{type(self.loss).__name__}({s})'
        if self.factor != 1:
            s = f'{self.factor:.1g} * {s}'
        return s

    __str_ = __repr__


class SumSimilarity(BaseSimilarity):
    """A sum of Similarity losses"""

    def __init__(self, similarities):
        """
        Parameters
        ----------
        similarities : sequence[Similarity]
        """
        def flatten_losses(similarities):
            if isinstance(similarities, Similarity):
                return [similarities]
            flat_similarities = []
            for similarity in similarities:
                flat_similarities += flatten_losses(similarity)
            return flat_similarities
        self.similarities = flatten_losses(similarities)

    @classmethod
    def sum(cls, losses):
        obj = SumSimilarity(losses)
        if len(obj) == 1:
            return obj[0]
        return obj

    def images(self):
        for loss in self.similarities:
            for image in loss.images():
                yield image

    def losses(self):
        for similarity in self.similarities:
            for loss in similarity.losses():
                yield loss

    def __len__(self):
        return len(self.similarities)

    def __iter__(self):
        for loss in self.similarities:
            yield loss

    def __getitem__(self, item):
        return self.similarities[item]

    def __setitem__(self, item, value):
        self.similarities[item] = value

    def reverse(self):
        return SumSimilarity([loss.reverse() for loss in self])

    def symmetrize(self):
        return SumSimilarity([*self, *self.reverse()]) * 0.5

    def __mul__(self, factor):
        return SumSimilarity([factor * loss for loss in self])

    def __rmul__(self, factor):
        return SumSimilarity([loss * factor for loss in self])

    def __imul__(self, factor):
        for loss in self:
            loss *= factor
        return self

    def __truediv__(self, factor):
        return SumSimilarity([factor / loss for loss in self])

    def __itruediv__(self, factor):
        for loss in self:
            loss /= factor
        return self

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        if isinstance(other, SumSimilarity):
            return SumSimilarity([*self, *other])
        elif isinstance(other, BaseSimilarity):
            return SumSimilarity([*self, other])
        raise TypeError(f'Cannot add a {type(self)} and a {type(other)}')

    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        if isinstance(other, SumSimilarity):
            return SumSimilarity([*other, *self])
        elif isinstance(other, BaseSimilarity):
            return SumSimilarity([other, *self])
        raise TypeError(f'Cannot add a {type(other)} and a {type(self)}')

    def __iadd__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        if isinstance(other, SumSimilarity):
            self.similarities += other.losses
        elif isinstance(other, BaseSimilarity):
            self.similarities.append(other)
        else:
            raise TypeError(f'Cannot add a {type(self)} and a {type(other)}')
        return self

    def __repr__(self):
        s = ' + '.join([repr(loss) for loss in self])
        return f'[{s}]'

    __str_ = __repr__


class Space:
    """An oriented lattice (shape and affine)"""
    def __init__(self, shape, affine=None, voxel_size=None):
        if affine is None:
            affine = affine_default(shape, voxel_size)
        self.shape = shape
        self.affine = affine

    def __repr__(self):
        vx = get_voxel_size(self.affine).tolist()
        return f'{type(self).__name__}(shape={self.shape}, vx={vx})'


class MeanSpace(Space):
    """Compute a mean space from a bunch of affine + shape"""
    def __init__(self, images, voxel_size=None, vx_unit='mm', pad=0, pad_unit='%'):
        mat, shape = mean_space(
            [image.affine for image in images],
            [image.shape for image in images],
            voxel_size=voxel_size, vx_unit=vx_unit,
            pad=pad, pad_unit=pad_unit)
        super().__init__(shape, mat)


class SpatialTensor:
    """Base class for tensors with an orientation"""

    def __init__(self, dat, affine=None, **backend):
        """
        Parameters
        ----------
        dat : (C, *spatial) tensor or MappedArray or SpatialTensor
            Image data
        affine : (D+1, D+1) tensor, optional
            Orientation matrix.
        dtype : torch.dtype, optional
        device : torch.device, optional
        """
        if isinstance(dat, io.MappedArray):
            if affine is None:
                affine = dat.affine
            dat = dat.fdata(rand=True, **backend)
        if isinstance(dat, SpatialTensor):
            if affine is None:
                affine = dat.affine
            dat = dat.dat
        self.dim = dat.dim() - 1
        self.dat = dat.to(**backend)
        if affine is None:
            affine = affine_default(self.shape, **get_backend(dat))
        self.affine = affine.to(self.dat.device)

    def to(self, *args, **kwargs):
        return copy.copy(self).to_(*args, **kwargs)

    def to_(self, *args, **kwargs):
        self.dat = self.dat.to(*args, **kwargs)
        self.affine = self.affine.to(*args, **kwargs)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def cuda_(self):
        return self.to_('cuda')

    def cpu_(self):
        return self.to_('cpu')

    voxel_size = property(lambda self: get_voxel_size(self.affine))
    shape = property(lambda self: self.dat.shape[-self.dim:])
    dtype = property(lambda self: self.dat.dtype)
    device = property(lambda self: self.dat.device)

    def _prm_as_str(self):
        s = [f'shape={list(self.shape)}']
        v = [f'{vx:.2g}' for vx in self.voxel_size.tolist()]
        v = ', '.join(v)
        s += [f'voxel_size=[{v}]']
        if self.dtype != torch.float32:
            s += [f'dtype={str(torch.int32).split(".")[-1]}']
        if self.device.type != 'cpu':
            s += [f'device={self.device.type}']
        return s

    def __repr__(self):
        s = ', '.join(self._prm_as_str())
        s = f'{self.__class__.__name__}({s})'
        return s

    __str__ = __repr__


class Image(SpatialTensor):
    """Data + Metadata (affine, boundary) of an Image"""
    def __init__(self, dat, affine=None, mask=None,
                 bound='dct2', extrapolate=False, preview=None):
        """
        Parameters
        ----------
        dat : (C, *spatial) tensor or MappedArray or SpatialTensor
            Image data
        affine : tensor, optional
            Orientation matrix
        mask : (C|1, *spatial) tensor, optional
            Mask of valid voxels
        bound : str, default='dct2'
            Boundary conditions for interpolation
        extrapolate : bool, default=True
            Whether to extrapolate out-of-bound data
        preview : (C|1, *spatial) tensor, optional
            Image used when plotting
        """
        super().__init__(dat, affine)
        self.bound = bound
        self.extrapolate = extrapolate
        if mask is not None:
            while mask.dim() < self.dim + 1:
                mask = mask[None]
        self.mask = mask
        if self.masked and mask.shape[-self.dim:] != self.shape:
            raise ValueError('Wrong shape for mask')
        if preview is not None:
            while preview.dim() < self.dim + 1:
                preview = preview[None]
        self._preview = preview
        if self.previewed and preview.shape[-self.dim:] != self.shape:
            raise ValueError('Wrong shape for preview')

    masked = property(lambda self: self.mask is not None)
    previewed = property(lambda self: self._preview is not None)
    preview = property(lambda self: self._preview if self.previewed else self.dat)

    @preview.setter
    def preview(self, x):
        self._preview = x

    def pull(self, grid, dat=True, mask=False, preview=False):
        """Sample the image at dense coordinates.

        Parameters
        ----------
        grid : (*spatial, dim) tensor or None
            Dense transformation field.
        dat : bool, default=True
            Return warped image
        mask : bool, default=False
            Return warped mask
        preview : bool, default=False
            Return warped preview image

        Returns
        -------
        warped, if `dat` : (C, *spatial) tensor
        warped_mask, if `mask` : (C|1, *spatial) tensor
        warpe_previewd, if `preview` : (C|1, *spatial) tensor

        """
        opt = dict(bound=self.bound, extrapolate=self.extrapolate)
        out = []
        img = None
        if dat or (preview and not self.previewed):
            img = interpol.pull(
                self.dat.movedim(0, -1), grid, **opt
            ).movedim(-1, 0)
            if dat:
                out += [img]
        if mask:
            msk = None
            if self.masked:
                msk = self.mask.to(self.dat.dtype)
                msk = interpol.pull(
                    msk.movedim(0, -1), grid, **opt
                ).movedim(-1, 0)
            out += [msk]
        if preview:
            if self.previewed:
                prv = self.preview.to(self.dat.dtype)
                prv = interpol.pull(
                    prv.movedim(0, -1), grid, **opt
                ).movedim(-1, 0)
            else:
                prv = img
            out += [prv]
        return tuple(out) if len(out) > 1 else out[0]

    def pull_grad(self, grid, rotate=False):
        """Sample the image gradients at dense coordinates.

        Parameters
        ----------
        grid : (*spatial, dim) tensor or None
            Dense transformation field.
        rotate : bool, default=False
            Rotate the gradients using the Jacobian of the transformation.

        Returns
        -------
        grad : (C, *spatial, dim) tensor

        """
        opt = dict(bound=self.bound, extrapolate=self.extrapolate)
        if grid is None:
            return self.grad()
        grad = interpol.pullgrad(
            self.dat.movedim(0, -1), grid, **opt
        ).movedim(-1, 0)
        if rotate:
            jac = flow_jacobian(sub_identity_grid(grid))
            jac = jac.transpose(-1, -2)
            grad = fastmath.matvec(jac, grad)
        return grad

    def grad(self):
        """Compute the image gradients in each voxel.
        Almost equivalent to `self.pull_grad(identity)`.

        Returns
        -------
        grad : (C, *spatial, dim) tensor

        """
        dims = list(range(-self.dim, 0))
        return diff(self.dat, dim=dims, bound=self.bound)

    def _prm_as_str(self):
        s = []
        if self.bound != 'dct2':
            s += [f'bound={self.bound}']
        if self.extrapolate:
            s += ['extrapolate=True']
        if self.masked:
            s += ['masked=True']
        s = super()._prm_as_str() + s
        return s


class ImageSequence(Image):
    """A list of images that can also be used as an image"""
    def __init__(self, dat, affine=None, dim=None, mask=None,
                 bound='dct2', extrapolate=False, **backend):
        # I don't call super().__init__() on purpose
        if torch.is_tensor(affine):
            affine = [affine] * len(dat)
        elif affine is None:
            affine = []
            for dat1 in dat:
                dim1 = dim or dat1.dim
                if callable(dim1):
                    dim1 = dim1()
                if hasattr(dat1, 'affine'):
                    aff1 = dat1.affine
                else:
                    shape1 = dat1.shape[-dim1:]
                    aff1 = affine_default(shape1, **get_backend(dat1))
                affine.append(aff1)
        affine = make_list(affine, len(dat))
        mask = make_list(mask, len(dat))
        self._dat = []
        for dat1, aff1, mask1 in zip(dat, affine, mask):
            if not isinstance(dat1, Image):
                dat1 = Image(dat1, aff1, mask=mask1,
                             bound=bound, extrapolate=extrapolate)
            self._dat.append(dat1)

    def __len__(self):
        return len(self._dat)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._dat[item]
        return ImageSequence(self._dat[item])

    def __iter__(self):
        for dat in self._dat:
            yield dat

    dim = property(lambda self: self[0].dim)
    shape = property(lambda self: self[0].shape)
    dat = property(lambda self: self[0].dat)
    affine = property(lambda self: self[0].affine)
    bound = property(lambda self: self[0].bound)
    extrapolate = property(lambda self: self[0].extrapolate)
    voxel_size = property(lambda self: self[0].voxel_size)
    mask = property(lambda self: self[0].mask)
    preview = property(lambda self: self[0].preview)

    def to(self, *args, **kwargs):
        return copy.deepcopy(self).to_(*args, **kwargs)

    def to_(self, *args, **kwargs):
        for dat in self:
            dat.to_(*args, **kwargs)
        return self

    def _prm_as_str(self):
        return [f'nb_images={len(self)}'] + super()._prm_as_str()


class ImagePyramid(ImageSequence):
    """
    Compute a multiscale image pyramid.
    This object can be used as an Image (in which case the highest
    resolution is returned) or as a list of Images.
    """

    def __init__(self, dat, levels=1, affine=None, dim=None,
                 mask=None, preview=None,
                 bound='dct2', extrapolate=False, method='gauss', **backend):
        """

        Parameters
        ----------
        dat : [list of] (..., *shape) tensor or Image
        levels : int or list[int] or range, default=0
            If an int, it is the number of levels.
            If a range or list, they are the indices of levels to compute.
            `0` is the native resolution, `1` is half of it, etc.
        affine : [list of] tensor, optional
        dim : int, optional
        bound : str, default='dct2'
        extrapolate : bool, default=True
        method : {'gauss', 'average', 'median', 'stride'}, default='gauss'
        """
        # I don't call super().__init__() on purpose
        self.method = method

        if isinstance(dat, Image):
            if affine is None:
                affine = dat.affine
            dim = dat.dim
            mask = dat.mask
            preview = getattr(dat, '_preview', None)
            bound = dat.bound
            extrapolate = dat.extrapolate
            dat = dat.dat
        if isinstance(dat, str):
            dat = io.map(dat)
        if isinstance(dat, io.MappedArray):
            if affine is None:
                affine = dat.affine
            dat = dat.fdata(rand=True, **backend)[None]

        if isinstance(levels, int):
            levels = range(levels)
        if isinstance(levels, range):
            levels = list(levels)

        if not isinstance(dat, (list, tuple)):
            dim = dim or dat.dim() - 1
            if not levels:
                raise ValueError('levels required to compute pyramid')
            if isinstance(levels, int):
                levels = range(1, levels+1)
            if affine is None:
                shape = dat.shape[-dim:]
                affine = affine_default(shape, **get_backend(dat[0]))
            dat, mask, preview = self._build_pyramid(
                dat, levels, method, dim, bound, mask, preview)
        dat = list(dat)
        if not mask:
            mask = [None] * len(dat)
        if not preview:
            preview = [None] * len(dat)

        if all(isinstance(d, Image) for d in dat):
            self._dat = dat
            return

        dim = dim or (dat[0].dim() - 1)
        if affine is None:
            shape = dat[0].shape[-dim:]
            affine = affine_default(shape, **get_backend(dat[0]))
        if not isinstance(affine, (list, tuple)):
            if not levels:
                raise ValueError('levels required to compute affine pyramid')
            shape = dat[0].shape[-dim:]
            affine = self._build_affine_pyramid(affine, shape, levels, method)
        self._dat = [Image(d, aff, mask=m, preview=p,
                           bound=bound, extrapolate=extrapolate)
                     for d, m, p, aff in zip(dat, mask, preview, affine)]

    def _prm_as_str(self):
        s = []
        if self.method[0] != 'g':
            s += [f'method={self.method}']
        return [f'nb_levels={len(self)}'] + super()._prm_as_str() + s

    def _build_pyramid(self, dat, levels, method, dim, bound,
                       mask=None, preview=None):
        levels = list(levels)
        indexed_levels = list(enumerate(levels))
        indexed_levels.sort(key=lambda x: x[1])
        nb_levels = max(levels)
        if mask is not None:
            mask = mask.to(dat.device)
        dats = [dat] * levels.count(0)
        masks = [mask] * levels.count(0)
        previews = [preview] * levels.count(0)
        if mask is not None:
            mask = mask.to(dat.dtype)
        if preview is not None:
            preview = preview.to(dat.dtype)
        for level in range(1, nb_levels+1):
            shape = dat.shape[-dim:]
            kernel_size = [min(2, s) for s in shape]
            if method[0] == 'g':  # gaussian pyramid
                # We assume the original data has a PSF of 1 input voxel.
                # We smooth by an additional 1-vx FWHM so that the data has a
                # PSF of 2 input voxels == 1 output voxel, then subsample.
                smooth_fn = lambda x: smooth(x, fwhm=1, stride=2,
                                             dim=dim, bound=bound)
            elif method[0] == 'a':  # average window
                smooth_fn = lambda x: pool(dim, x, kernel_size=kernel_size,
                                           stride=2, reduction='mean')
            elif method[0] == 'm':  # median window
                smooth_fn = lambda x: pool(dim, x, kernel_size=kernel_size,
                                           stride=2, reduction='median')
            elif method[0] == 's':  # strides
                slicer = [slice(None, None, 2)] * dim
                smooth_fn = lambda x: x[(Ellipsis, *slicer)]
            else:
                raise ValueError(method)
            dat = smooth_fn(dat)
            if mask is not None:
                mask = smooth_fn(mask)
            if preview is not None:
                preview = smooth_fn(preview)
            dats += [dat] * levels.count(level)
            masks += [mask] * levels.count(level)
            previews += [preview] * levels.count(level)
        reordered_dats = [None] * len(levels)
        reordered_masks = [None] * len(levels)
        reordered_previews = [None] * len(levels)
        for (i, level), dat, mask, preview \
                in zip(indexed_levels, dats, masks, previews):
            reordered_dats[i] = dat
            reordered_masks[i] = mask
            reordered_previews[i] = preview
        return reordered_dats, reordered_masks, reordered_previews

    def _build_affine_pyramid(self, affine, shape, levels, method):
        levels = list(levels)
        indexed_levels = list(enumerate(levels))
        indexed_levels.sort(key=lambda x: x[1])
        nb_levels = max(levels)
        affines = [affine] * levels.count(0)
        for level in range(1, nb_levels+1):
            if method[0] == 's':  # stride pyramid
                slicer = (slice(None, None, 2),) * len(shape)
                affine, shape = affine_sub(affine, shape, slicer)
            else:  # conv pyramid
                if method[0] == 'g':  # gaussian pyramid
                    padding = 'auto'
                    kernel = 3
                else:  # moving window
                    padding = 0
                    kernel = [min(2, s) for s in shape]
                affine, shape = affine_conv(affine, shape, kernel, 2,
                                            padding=padding)
            affines += [affine] * levels.count(level)
        reordered_affines = [None] * len(levels)
        for (i, level), affine in zip(indexed_levels, affines):
            reordered_affines[i] = affine
        return reordered_affines


class SpatialTransform:

    def _prm_as_str(self):
        return []

    def __repr__(self):
        s = ', '.join(self._prm_as_str())
        s = f'{self.__class__.__name__}({s})'
        return s

    def __str__(self):
        return repr(self)


class AffineTransform(SpatialTransform):
    pass


class DenseTransform(SpatialTransform):
    pass


class Displacement(SpatialTensor, DenseTransform):
    """
    Data + Metadata (affine) of a displacement or velocity field (in voxels)
    """

    def __init__(self, dat, affine=None, **backend):
        """
        Parameters
        ----------
        dat : (*spatial, dim) tensor or list[int]
            Pre-allocated displacement field, or its shape.
        affine : tensor, optional
            Orientation matrix
        """
        if isinstance(dat, (list, tuple)):
            shape = dat
            dim = len(shape)
            dat = torch.zeros([*shape, dim], **backend)
        super().__init__(dat, affine)

    @property
    def shape(self):
        return self.dat.shape[-self.dim-1:-1]

    _upkwargs = dict(
        interpolation=3,
        bound='dft',
        anchor='c',
        extrapolate=True,
        type='displacement',
    )

    def downsample_(self, factor=2, **kwargs):
        return self.upsample_(1/factor, **kwargs)

    def downsample(self, factor=2, **kwargs):
        return self.upsample(1/factor, **kwargs)

    def upsample_(self, factor=2, **kwargs):
        kwargs.update(self._upkwargs)
        inshape = self.dat.shape[-self.dim:]
        self.dat = resize_flow(self.dat, factor, **kwargs)
        self.affine = resize_affine(
            self.affine, inshape, factor=factor, anchor=kwargs['anchor'])
        return self

    def upsample(self, factor=2, **kwargs):
        kwargs.update(self._upkwargs)
        inshape = self.dat.shape[-self.dim:]
        dat = resize_flow(self.dat, factor, **kwargs)
        aff = resize_affine(
            self.affine, inshape, factor=factor, anchor=kwargs['anchor'])
        return type(self)(dat, aff, dim=self.dim)


class LogAffine(AffineTransform):
    """Data + Metadata (basis) of a "Lie" affine matrix"""
    def __init__(self, dat=None, basis=None, dim=3, **backend):
        """
        Parameters
        ----------
        [dat : tensor, optional]
            Pre-allocated log-affine
        basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
            Name of an Affine basis
        dim : int, default=3
            Number of spatial dimensions
        **backend
        """
        if isinstance(dat, str):
            if basis is not None:
                raise ValueError('`basis` provided but `dat` looks like '
                                 'a basis.')
            basis, dat = dat, None
        elif basis is None:
            raise ValueError('A basis should be provided')
        self._basis = None
        if torch.is_tensor(basis):
            dim = basis.shape[-1] - 1
        self.dim = dim
        self.basis = basis
        if dat is None:
            dat = torch.zeros(affine_basis_size(basis, dim), **backend)
        self.dat = dat
        self._cache = None
        self._icache = None

    dtype = property(lambda self: self.dat.dtype if self.dat is not None else None)
    device = property(lambda self: self.dat.device if self.dat is not None else None)

    def to(self, *args, **kwargs):
        return copy.copy(self).to_(*args, **kwargs)

    def to_(self, *args, **kwargs):
        if self.dat is not None:
            self.dat = self.dat.to(*args, **kwargs)
        if self._basis is not None:
            self._basis = self._basis.to(*args, **kwargs)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def cuda_(self):
        return self.to_('cuda')

    def cpu_(self):
        return self.to_('cpu')

    @property
    def basis(self):
        if self._basis is None:
            begin_rot = self.dim
            begin_scl = begin_rot + (self.dim*(self.dim-1))//2
            if self._basis_name == 'affine':
                end_scl = begin_scl + self.dim
            else:
                end_scl = begin_scl + 1
            self._basis = affine_basis(self._basis_name, self.dim,
                                       device=self.dat.device,
                                       dtype=torch.double)
            self._basis[begin_rot:begin_scl] /= 40  # rotations ~ 1 deg
            self._basis[begin_scl:end_scl] /= 40    # scalings  ~ 1 pct
            self._basis[end_scl:] /= 40             # shears
        return self._basis

    @basis.setter
    def basis(self, x):
        self._basis_name = x
        self._basis = None

    def switch_basis(self, basis):
        """Return a transform with the same log-parameters but a different basis

        Parameters
        ----------
        basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
            Name of an Affine basis

        Returns
        -------
        affine : LogAffine
            LogAffine with a different basis

        """
        dat = None
        dim = self.dim
        if self._basis_name is not None and self.dat is not None:
            new_basis = basis[:2].lower()
            old_basis = self._basis_name[:2].lower()
            if new_basis[0] == 't':
                dat = self.dat[:dim].clone()
            elif new_basis == 'ro':
                nb_prm = dim*(dim-1)//2
                if old_basis == 'ro':
                    dat = self.dat.clone()
                elif len(self.dat) >= dim + nb_prm:
                    dat = self.dat[dim:dim+nb_prm].clone()
                else:
                    dat = self.dat.new_zeros([nb_prm])
                    dat1 = self.dat[dim:dim+nb_prm]
                    dat[:len(dat1)] = dat1
            elif new_basis[0] == 'r':
                nb_prm = dim + dim*(dim-1)//2
                if old_basis == 'ro':
                    dat = self.dat.new_zeros([nb_prm])
                    dat[dim:] = self.dat
                elif len(self.dat) >= dim + nb_prm:
                    dat = self.dat[:nb_prm].clone()
                else:
                    dat = self.dat.new_zeros([nb_prm])
                    dat[:len(self.dat)] = self.dat
            elif new_basis[0] == 's':
                nb_prm = dim + dim*(dim-1)//2 + 1
                dat = self.dat.new_zeros([nb_prm])
                if old_basis == 'ro':
                    dat[dim:-1] = self.dat
                elif len(self.dat) >= dim + nb_prm:
                    # old_basis is full affine
                    dat[:nb_prm-1] = self.dat[:nb_prm-1]
                    dat[-1] = self.dat[nb_prm-1:nb_prm+2].mean()
                else:
                    dat[:len(self.dat)] = self.dat
            else:
                nb_prm = dim*(dim+1)
                dat = self.dat.new_zeros([nb_prm])
                if old_basis == 'ro':
                    dat[dim:dim+dim*(dim-1)//2] = self.dat
                elif old_basis[0] == 's':
                    dat[:len(self.dat)-1] = self.dat[:-1]
                    dat[len(self.dat)-1:len(self.dat)+dim-1] = self.dat[-1]
                else:
                    dat[:len(self.dat)] = self.dat
        return LogAffine(dat, basis, self.dim)

    def clear_cache(self):
        self._cache = None
        self._icache = None

    def exp(self, q=None, grad=False, cache_result=False, recompute=True):
        if q is None:
            q = self.dat
        if grad:
            recompute = True
        if recompute or getattr(self, '_cache') is None:
            aff = fastmath.expm_derivatives(
                q.double(), self.basis, grad_X=grad
            )
            aff = (list(map(lambda x: x.to(q.dtype), aff))
                   if isinstance(aff, (list, tuple)) else aff.to(q.dtype))
        else:
            aff = self._cache
        if cache_result:
            self._cache = aff[0] if grad else aff
        return aff

    def iexp(self, q=None, grad=False, cache_result=False, recompute=True):
        if q is None:
            q = self.dat
        if grad:
            recompute = True
        if recompute or self._cache is None:
            iaff = fastmath.expm_derivatives(
                -q.double(), self.basis, grad_X=grad
            )
            iaff = (list(map(lambda x: x.to(q.dtype), iaff))
                    if isinstance(iaff, (list, tuple)) else iaff.to(q.dtype))
        else:
            iaff = self._cache
        if cache_result:
            self._cache = iaff[0] if grad else iaff
        return iaff

    def exp2(self, q=None, grad=False, cache_result=False, recompute=True):
        if grad:
            recompute = True
        if grad:
            a, g = self.exp(q, True)
            ia, ig = self.iexp(q, True)
            return a, ia, g, ig
        else:
            return (self.exp(q, cache_result=cache_result, recompute=recompute),
                    self.iexp(q, cache_result=cache_result, recompute=recompute))

    def _prm_as_str(self):
        return [f'shape={list(self.dat.shape)}', f'dim={self.dim}']

    def __repr__(self):
        s = [f'shape={list(self.dat.shape)}', f'dim={self.dim}']
        s = ', '.join(s)
        return f'{self.__class__.__name__}({s})'

    __str__ = __repr__


class LogAffine2d(LogAffine):
    """2d affine transform in a 3D world space"""
    def __init__(self, dat=None, basis=None, rotation=None,
                 plane=0, dim=3, **backend):
        """
        Parameters
        ----------
        [dat : tensor, optional]
            Pre-allocated log-affine
        basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
            Name of an Affine basis
        rotation : tensor, default=identity
            Rotation matrix from 2d space to 3d space
        dim : int, default=3
            Number of spatial dimensions
        **backend
        """
        if dat is None:
            dat = torch.zeros(affine_basis_size(basis, 2), **backend)
        super().__init__(dat, basis, dim, **backend)
        if rotation is None:
            rotation = torch.eye(self.dim + 1, **backend)
        self.rotation = rotation
        if isinstance(plane, str):
            plane = plane.upper()
            plane = (0 if plane in 'RL' else 1 if plane in 'AP' else 2)
        self.plane = plane

    def _make_basis_2d(self):
        if self._basis is not None:
            if self._basis_name[0].lower() == 't':
                if self.plane == 0:
                    idx = [1, 2]
                elif self.plane == 1:
                    idx = [0, 2]
                else:
                    idx = [0, 1]
            elif self._basis_name[0].lower() in 'rs':
                if self.plane == 0:
                    idx = [1, 2, 5]
                elif self.plane == 1:
                    idx = [0, 2, 4]
                else:
                    idx = [0, 1, 3]
                if self._basis_name[0].lower() == 's':
                    idx += [6]
            else:
                if self.plane == 0:
                    idx = [1, 2, 5, 7, 8, 11]
                elif self.plane == 1:
                    idx = [0, 2, 1, 6, 8, 10]
                else:
                    idx = [0, 1, 3, 6, 7, 9]

            self._basis = self._basis[idx]
            if self._basis_name[0].lower() == 's':
                if self.plane == 0:
                    idx = 0
                elif self.plane == 1:
                    idx = 1
                else:
                    idx = 2
                self._basis[3][idx, idx] = 0

    @property
    def basis(self):
        make_basis_2d = self._basis is None
        _ = super().basis
        if make_basis_2d:
            self._make_basis_2d()
        return self._basis

    @basis.setter
    def basis(self, x):
        self._basis_name = x
        self._basis = None

    def switch_basis(self, basis):
        """Return a transform with the same log-parameters but a different basis

        Parameters
        ----------
        basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
            Name of an Affine basis

        Returns
        -------
        affine : LogAffine2d
            LogAffine2d with a different basis

        """
        dat = None
        dim = 2
        if self._basis_name is not None and self.dat is not None:
            new_basis = basis[:2].lower()
            old_basis = self._basis_name[:2].lower()
            if new_basis[0] == 't':
                dat = self.dat[:dim].clone()
            elif new_basis == 'ro':
                nb_prm = dim*(dim-1)//2
                if old_basis == 'ro':
                    dat = self.dat.clone()
                elif len(self.dat) >= dim + nb_prm:
                    dat = self.dat[dim:dim+nb_prm].clone()
                else:
                    dat = self.dat.new_zeros([nb_prm])
                    dat1 = self.dat[dim:dim+nb_prm]
                    dat[:len(dat1)] = dat1
            elif new_basis[0] == 'r':
                nb_prm = dim + dim*(dim-1)//2
                if old_basis == 'ro':
                    dat = self.dat.new_zeros([nb_prm])
                    dat[dim:] = self.dat
                elif len(self.dat) >= dim + nb_prm:
                    dat = self.dat[:nb_prm].clone()
                else:
                    dat = self.dat.new_zeros([nb_prm])
                    dat[:len(dat)] = dat
            elif new_basis[0] == 's':
                nb_prm = dim + dim*(dim-1)//2 + 1
                dat = self.dat.new_zeros([nb_prm])
                if old_basis == 'ro':
                    dat[dim:-1] = self.dat
                elif len(self.dat) >= dim + nb_prm:
                    # old_basis is full affine
                    dat[:nb_prm-1] = self.dat[:nb_prm-1]
                    dat[-1] = self.dat[nb_prm-1:nb_prm+2].mean()
                else:
                    dat[:len(dat)] = dat
            else:
                nb_prm = dim*(dim+1)
                dat = self.dat.new_zeros([nb_prm])
                if old_basis == 'ro':
                    dat[dim:dim+dim*(dim-1)//2] = self.dat
                elif old_basis[0] == 's':
                    dat[:len(self.dat)-1] = self.dat[:-1]
                    dat[len(self.dat)-1:] = self.dat[-1]
                else:
                    dat[:len(dat)] = dat
        return LogAffine2d(dat, basis, dim=self.dim,
                           plane=self.plane, rotation=self.rotation)

    def exp(self, q=None, grad=False, cache_result=False, recompute=True):
        aff = super().exp(q, grad, cache_result, recompute)
        if grad:
            aff, gaff = aff
        dtype = aff.dtype
        aff = aff.double()
        rot = self.rotation.to(aff)
        aff = (rot @ aff @ rot.T).to(dtype)
        if grad:
            gaff = gaff.double()
            gaff = rot.matmul(gaff).matmul(rot.T).to(dtype)
            return aff, gaff
        return aff

    def iexp(self, q=None, grad=False, cache_result=False, recompute=True):
        aff = super().iexp(q, grad, cache_result, recompute)
        if grad:
            aff, gaff = aff
        dtype = aff.dtype
        aff = aff.double()
        rot = self.rotation.to(aff)
        aff = (rot @ aff @ rot.T).to(dtype)
        if grad:
            gaff = gaff.double()
            gaff = rot.matmul(gaff).matmul(rot.T).to(dtype)
            return aff, gaff
        return aff

    def exp2(self, q=None, grad=False, cache_result=False, recompute=True):
        aff = super().exp2(q, grad, cache_result, recompute)
        if grad:
            aff, iaff, gaff, giaff = aff
        else:
            aff, iaff = aff
        dtype = aff.dtype
        aff = aff.double()
        iaff = iaff.double()
        rot = self.rotation.to(aff)
        aff = (rot @ aff @ rot.T).to(dtype)
        iaff = (rot @ iaff @ rot.T).to(dtype)
        if grad:
            gaff = gaff.double()
            giaff = giaff.double()
            gaff = rot.matmul(gaff).matmul(rot.T).to(dtype)
            giaff = rot.matmul(giaff).matmul(rot.T).to(dtype)
            return aff, iaff, gaff, giaff
        return aff, iaff

    def _prm_as_str(self):
        s = super()._prm_as_str()
        s += [f'plane={self.plane}']
        return s


def _rotate_grad(grad, aff=None, dense=None):
    """Rotate grad by the jacobian of `aff o dense`.
    grad : (..., dim) tensor       Spatial gradients
    aff : (dim+1, dim+1) tensor    Affine matrix
    dense : (..., dim) tensor      Dense vox2vox displacement field
    returns : (..., dim) tensor    Rotated gradients.
    """
    if aff is None and dense is None:
        return grad
    dim = grad.shape[-1]
    if dense is not None:
        jac = flow_jacobian(dense)
        if aff is not None:
            jac = torch.matmul(aff[:dim, :dim], jac)
    else:
        jac = aff[:dim, :dim]
    grad = fastmath.matvec(jac.transpose(-1, -2), grad)
    return grad


class TransformationModel:
    def parameters(self):
        raise NotImplementedError

    def _prm_as_str(self):
        return []

    def __repr__(self):
        s = ', '.join(self._prm_as_str())
        s = f'{self.__class__.__name__}({s})'
        return s

    def __str__(self):
        return repr(self)


class NonLinModel(TransformationModel):
    """Base class for non-linear deformations."""
    # TODO: currently tailored for dense fields but could be adapted
    #   to spline-encoded deformations (or velocities)

    @classmethod
    def subclass(cls, model):
        model = model.lower()
        if model == 'shoot':
            return ShootModel
        elif model == 'svf':
            return SVFModel
        elif model == 'smalldef':
            return SmallDefModel
        else:
            raise ValueError('unknown:', model)

    def __new__(cls, *args, **kwargs):
        if cls is NonLinModel:
            if args:
                model, *args = args
            else:
                model = kwargs.pop('model', None)
            if isinstance(model, NonLinModel):
                return model
            elif model.lower() == 'shoot':
                return ShootModel(*args, **kwargs)
            elif model.lower() == 'svf':
                return SVFModel(*args, **kwargs)
            elif model.lower() == 'smalldef':
                return SmallDefModel(*args, **kwargs)
            else:
                raise ValueError('unknown:', model)
        else:
            obj =  object.__new__(cls)
            obj.__init__(*args, **kwargs)
            return obj

    def __init__(self, dat=None, factor=1, penalty=None,
                 steps=8, kernel=None, **backend):
        """

        Parameters
        ----------
        dat : Displacement or tensor or list[int], optional
            Pre-allocated displacement or its shape
        factor : float, default=1
            Regularization factor
        penalty : dict(absolute, membrane, bending, lame), optional
            Regularization factor for each component.
            Informed values are used by default.
        steps : int, default=8
            Number of integration steps.
        kernel : tensor, optional
            Pre-computed Greens kernel.
        **backend
        """
        super().__init__()
        self.factor = factor
        self.penalty = dict(penalty or regutils.defaults_velocity())
        if factor is not None:
            self.penalty['factor'] = factor
        self.penalty.pop('voxel_size', None)
        self.steps = steps
        self.kernel = kernel
        if dat is not None:
            self.set_dat(dat, **backend)
        else:
            self.dat = None
        self._cache = None
        self._icache = None

    affine = property(lambda self: self.dat.affine if self.dat is not None else None)
    shape = property(lambda self: self.dat.shape if self.dat is not None else None)
    dim = property(lambda self: self.dat.dim if self.dat is not None else None)
    voxel_size = property(lambda self: self.dat.voxel_size if self.dat is not None else None)

    def to(self, *args, **kwargs):
        return copy.copy(self).to_(*args, **kwargs)

    def to_(self, *args, **kwargs):
        if self.dat is not None:
            self.dat.to_(*args, **kwargs)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def cuda_(self):
        return self.to_('cuda')

    def cpu_(self):
        return self.to_('cpu')

    def parameters(self):
        if not self.dat:
            return None
        return self.dat.dat

    def set_dat(self, dat, affine=None, **backend):
        if isinstance(dat, io.MappedArray):
            if affine is None:
                affine = dat.affine
            dat = dat.fdata()
        self.dat = Displacement(dat, affine=affine, **backend)
        return self

    def set_kernel(self, kernel=None):
        if kernel is None:
            penalty = dict(self.penalty)
            factor = penalty.pop('factor')
            kernel = greens(self.shape, **penalty,
                            factor=factor / prod(self.shape),
                            voxel_size=self.voxel_size,
                            **get_backend(self.dat))
        self.kernel = kernel
        return self

    def reset_kernel(self, kernel=None):
        return self.set_kernel(kernel)  # backward compatibility

    def clear_cache(self):
        self._cache = None
        self._icache = None

    def regulariser(self, v=None):
        if v is None:
            v = self.dat
        penalty = dict(**self.penalty, voxel_size=self.voxel_size)
        factor = penalty.pop('factor', 1) / prod(self.shape)
        return flow_penalty(v, **penalty).mul_(factor)

    def greens_apply(self, m):
        return greens_apply(m, self.kernel, self.factor, self.voxel_size)

    def downsample_(self, factor=2, **kwargs):
        self.clear_cache()
        self.dat.downsample_(factor, **kwargs)
        if self.kernel is not None:
            self.set_kernel()
        return self

    def downsample(self, factor=2, **kwargs):
        dat = self.dat.downsample(factor, **kwargs)
        obj = type(self)(dat, self.factor, self.penalty, self.steps)
        if self.kernel is not None:
            obj.set_kernel()
        return obj

    def upsample_(self, factor=2, **kwargs):
        self.clear_cache()
        self.dat.upsample_(factor, **kwargs)
        if self.kernel is not None:
            self.set_kernel()
        return self

    def upsample(self, factor=2, **kwargs):
        dat = self.dat.upsample(factor, **kwargs)
        obj = type(self)(dat, self.factor, self.penalty, self.steps)
        if self.kernel is not None:
            obj.set_kernel()
        return obj

    def _prm_as_str(self):
        s = []
        if self.dat is not None:
            s += self.dat._prm_as_str()
        else:
            s += ['<uninitialized>']
        s += [f'{key}={value}' for key, value in self.penalty.items()]
        return s


class ShootModel(NonLinModel):
    """Initial velocity exponentiated by geodesic shooting"""

    model = 'shoot'

    def set_dat(self, dat, affine=None, **backend):
        super().set_dat(dat, affine, **backend)
        self.reset_kernel()
        return self

    def exp(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate forward transform"""
        if v is None:
            v = self.dat.dat
        if recompute or self._cache is None:
            grid = shoot(v, self.kernel, steps=self.steps,
                         factor=self.factor / prod(self.shape),
                         voxel_size=self.voxel_size, **self.penalty,
                         displacement=True)
        else:
            grid = self._cache
        if cache_result:
            self._cache = grid
        if jacobian:
            jac = flow_jacobian(grid)
            if add_identity:
                grid = add_identity_grid(grid)
            return grid, jac
        else:
            if add_identity:
                grid = add_identity_grid(grid)
            return grid

    def iexp(self, v=None, jacobian=False, add_identity=False,
             cache_result=False, recompute=True):
        """Exponentiate inverse transform"""
        if v is None:
            v = self.dat.dat
        if recompute or self._icache is None:
            _, grid = shoot(
                v, self.kernel,
                steps=self.steps,
                factor=self.factor / prod(self.shape),
                voxel_size=self.voxel_size, **self.penalty,
                return_inverse=True,
                displacement=True,
            )
            if cache_result:
                self._icache = grid
        else:
            grid = self._icache
        if jacobian:
            jac = flow_jacobian(grid)
            if add_identity:
                grid = add_identity_grid(grid)
            return grid, jac
        else:
            if add_identity:
                grid = add_identity_grid(grid)
            return grid

    def exp2(self, v=None, jacobian=False, add_identity=False,
             cache_result=False, recompute=True):
        """Exponentiate both forward and inverse transforms"""
        if v is None:
            v = self.dat.dat
        if recompute or self._cache is None or self._icache is None:
            grid, igrid = shoot(
                v, self.kernel,
                steps=self.steps,
                factor=self.factor / prod(self.shape),
                voxel_size=self.voxel_size, **self.penalty,
                return_inverse=True,
                displacement=True,
            )
            if cache_result:
                self._cache = grid
                self._icache = igrid
        else:
            grid = self._cache
            igrid = self._icache
        if jacobian:
            jac = flow_jacobian(grid)
            ijac = flow_jacobian(igrid)
            if add_identity:
                grid = add_identity_grid(grid)
                igrid = add_identity_grid(igrid)
            return grid, igrid, jac, ijac
        else:
            if add_identity:
                grid = add_identity_grid(grid)
                igrid = add_identity_grid(igrid)
            return grid, igrid

    def propagate_grad(self, g, h, moving, phi, left=None, right=None, inv=False):
        """Convert derivatives wrt warped image in loss space to
        to derivatives wrt parameters
        parameters:
            g (tensor) : gradient wrt warped image
            h (tensor) : hessian wrt warped image
            moving (Image) : moving image
            phi (tensor) : dense (exponentiated) displacement field
            left (matrix) : left affine
            right (matrix) : right affine
            inv (bool) : whether we're in a backward symmetric pass
        returns:
            g (tensor) : pushed gradient
            h (tensor) : pushed hessian
            gmu (tensor) : rotated spatial gradients
        """
        # build bits of warp
        dim = phi.shape[-1]
        fixed_shape = g.shape[-dim:]
        moving_shape = moving.shape
        if inv:
            # differentiate wrt δ in: Left o (Id + D) o (Id + δ) o Right
            if right is not None:
                right = affine_grid(right, fixed_shape)
            g = regutils.smart_push(g, right, shape=self.shape)
            h = regutils.smart_push(h, right, shape=self.shape)
            del right

            phi_left = identity_grid(self.shape, **get_backend(phi))
            phi_left += phi
            if left is not None:
                phi_left = affine_matvec(left, phi_left)
            mugrad = moving.pull_grad(phi_left, rotate=False)
            del phi_left

            mugrad = _rotate_grad(mugrad, left, phi)

        else:
            # differentiate wrt δ in: Left o (Id - δ) o (Id + D) o Right
            if right is not None:
                right = affine_grid(right, fixed_shape)
                phi_right = regutils.smart_pull_grid(phi, right)
                phi_right += right
            else:
                phi_right = identity_grid(self.shape, **get_backend(phi))
                phi_right += phi

            g = g.neg_()
            g = regutils.smart_push(g, phi_right, shape=self.shape)
            h = regutils.smart_push(h, phi_right, shape=self.shape)
            del phi_right

            if left is not None:
                grid_left = affine_grid(left, self.shape)
            else:
                grid_left = left
            mugrad = moving.pull_grad(grid_left)
            del grid_left
            mugrad = _rotate_grad(mugrad, left)

        return g, h, mugrad


class SVFModel(NonLinModel):
    """Stationary velocity field exponentiated by scaling and squaring."""

    model = 'svf'

    def exp(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate forward transform"""
        if v is None:
            v = self.dat.dat
        if jacobian:
            recompute = True
        if recompute or self._cache is None:
            grid = exp_forward(
                v,
                steps=self.steps,
                jacobian=jacobian,
                displacement=True,
            )
            if cache_result:
                self._cache = grid[0] if jacobian else grid
        else:
            grid = self._cache
        if add_identity:
            if jacobian:
                grid = (add_identity_grid(grid[0]), grid[1])
            else:
                grid = add_identity_grid(grid)
        return grid

    def iexp(self, v=None, jacobian=False, add_identity=False,
             cache_result=False, recompute=True):
        """Exponentiate inverse transform"""
        if v is None:
            v = self.dat.dat
        if jacobian:
            recompute = True
        if recompute or self._icache is None:
            igrid = exp_forward(
                v,
                steps=self.steps,
                jacobian=jacobian,
                inverse=True,
                displacement=True,
            )
            if cache_result:
                self._icache = igrid[0] if jacobian else igrid
        else:
            igrid = self._icache
        if add_identity:
            if jacobian:
                igrid = (add_identity_grid(igrid[0]), igrid[1])
            else:
                igrid = add_identity_grid(igrid)
        return igrid

    def exp2(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate both forward and inverse transforms"""
        if v is None:
            v = self.dat.dat
        if jacobian:
            grid, jac = exp_forward(
                v,
                steps=self.steps,
                jacobian=True,
                displacement=True,
            )
            igrid, ijac = exp_forward(
                v,
                steps=self.steps,
                jacobian=True,
                inverse=True,
                displacement=True,
            )
            if cache_result:
                self._cache = grid
                self._icache = igrid
            if add_identity:
                grid = add_identity_grid(grid)
                igrid = add_identity_grid(igrid)
            return grid, igrid, jac, ijac
        else:
            if recompute or self._cache is None:
                grid = exp_forward(
                    v,
                    steps=self.steps,
                    jacobian=False,
                    displacement=True,
                )
            else:
                grid = self._cache
            if recompute or self._icache is None:
                igrid = exp_forward(
                    v,
                    steps=self.steps,
                    jacobian=False,
                    inverse=True,
                    displacement=True,
                )
            else:
                igrid = self._icache
            if cache_result:
                self._cache = grid
                self._icache = igrid
            grid = add_identity_grid(grid)
            igrid = add_identity_grid(igrid)
            return grid, igrid

    def propagate_grad(self, g, h, moving, phi, left=None, right=None, inv=False):
        """Convert derivatives wrt warped image in loss space to
        to derivatives wrt parameters
        parameters:
            g (tensor) : gradient wrt warped image
            h (tensor) : hessian wrt warped image
            moving (Image) : moving image
            phi (tensor) : dense (exponentiated) displacement field
            left (matrix) : left affine
            right (matrix) : right affine
            inv (bool) : whether we're in a backward symmetric pass
        returns:
            g (tensor) : pushed gradient
            h (tensor) : pushed hessian
            gmu (tensor) : rotated spatial gradients
        """
        if inv:
            g = g.neg_()

        # build bits of warp
        dim = phi.shape[-1]
        fixed_shape = g.shape[-dim:]
        moving_shape = moving.shape

        # differentiate wrt δ in: Left o Phi o (Id + δ) o Right
        # we'll then propagate them through Phi by scaling and squaring
        if right is not None:
            right = affine_grid(right, fixed_shape)
        g = regutils.smart_push(g, right, shape=self.shape)
        h = regutils.smart_push(h, right, shape=self.shape)
        del right

        phi_left = identity_grid(self.shape, **get_backend(phi))
        phi_left += phi
        if left is not None:
            phi_left = affine_matvec(left, phi_left)
        mugrad = moving.pull_grad(phi_left, rotate=False)
        del phi_left

        mugrad = _rotate_grad(mugrad, left, phi)

        return g, h, mugrad


class SmallDefModel(NonLinModel):
    """Dense displacement field."""

    model = 'smalldef'

    def exp(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate forward transform"""
        if v is None:
            v = self.dat.dat
        grid = v
        if jacobian:
            jac = flow_jacobian(grid)
        if add_identity:
            grid = add_identity_grid(grid)
        return (grid, jac) if jacobian else grid

    def iexp(self, v=None, jacobian=False, add_identity=False,
             cache_result=False, recompute=True):
        """Exponentiate inverse transform"""
        if v is None:
            v = self.dat.dat
        if recompute or self._icache is None:
            grid = flow_inv(v, type='disp', **self.penalty)
        else:
            grid = self._icache
        if cache_result:
            self._icache = grid
        if jacobian:
            jac = flow_jacobian(grid)
        if add_identity:
            grid = add_identity_grid(grid)
        return (grid, jac) if jacobian else grid

    def exp2(self, v=None, jacobian=False, add_identity=False,
            cache_result=False, recompute=True):
        """Exponentiate both forward and inverse transforms"""
        if v is None:
            v = self.dat.dat
        grid = v
        if recompute or self._icache is None:
            igrid = flow_inv(v, type='disp', **self.penalty)
        else:
            igrid = self._icache
        if cache_result:
            self._icache = igrid
        if jacobian:
            jac = flow_jacobian(grid)
            ijac = flow_jacobian(igrid)
        if add_identity:
            grid = add_identity_grid(grid)
            igrid = add_identity_grid(igrid)
        return (grid, igrid, jac, ijac) if jacobian else (grid, igrid)

    def propagate_grad(self, g, h, moving, phi, left=None, right=None, inv=False):
        """Convert derivatives wrt warped image in loss space to
        to derivatives wrt parameters
        parameters:
            g (tensor) : gradient wrt warped image
            h (tensor) : hessian wrt warped image
            moving (Image) : moving image
            phi (tensor) : dense (exponentiated) displacement field
            left (matrix) : left affine
            right (matrix) : right affine
            inv (bool) : whether we're in a backward symmetric pass
        returns:
            g (tensor) : pushed gradient
            h (tensor) : pushed hessian
            gmu (tensor) : rotated spatial gradients
        """
        if inv:
            g = g.neg_()

        # build bits of warp
        dim = phi.shape[-1]
        fixed_shape = g.shape[-dim:]

        # differentiate wrt δ in: Left o (Id + D + δ) o Right
        # (If `inv`, we assume that inv(Id + D) = Id - D)
        if right is not None:
            right = affine_grid(right, fixed_shape)
        g = regutils.smart_push(g, right, shape=self.shape)
        h = regutils.smart_push(h, right, shape=self.shape)
        del right

        phi_left = identity_grid(self.shape, **get_backend(phi))
        phi_left += phi
        if left is not None:
            phi_left = affine_matvec(left, phi_left)
        mugrad = moving.pull_grad(phi_left, rotate=False)
        del phi_left

        mugrad = _rotate_grad(mugrad, left)
        return g, h, mugrad


class Nonlin2dModel(NonLinModel):
    """A 3D Nonlinear field pointing along a 2D direction"""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(Nonlin2dModel)
        cls.__init__(obj, *args, **kwargs)
        return obj

    def __init__(self, model, plane, ref_affine=None, *args, **kwargs):
        if isinstance(plane, str):
            plane = plane.upper()
            plane = (0 if plane in 'RL' else 1 if plane in 'AP' else 2)
        self.plane = plane
        if ref_affine is None:
            rot = None
        else:
            rot = ref_affine.clone().double()
            rot[:-1, -1] = 0
            rot[:-1, :-1] /= rot[:-1, :-1].square().sum(0, keepdim=True).sqrt()
        self.rotation = rot
        self._model = model

    model = property(lambda self: self._model.model)
    factor = property(lambda self: self._model.factor)
    penalty = property(lambda self: self._model.penalty)
    steps = property(lambda self: self._model.steps)
    kernel = property(lambda self: self._model.kernel)
    dat = property(lambda self: self._model.dat)
    affine = property(lambda self: self._model.affine)
    shape = property(lambda self: self._model.shape)
    dim = property(lambda self: self._model.dim)
    voxel_size = property(lambda self: self._model.voxel_size)

    def exp(self, *args, **kwargs):
        return self._model.exp(*args, **kwargs)

    def iexp(self, *args, **kwargs):
        return self._model.iexp(*args, **kwargs)

    def exp2(self, *args, **kwargs):
        return self._model.exp2(*args, **kwargs)

    def propagate_grad(self, g, h, moving, phi, left=None, right=None, inv=False):
        g, h, mugrad = self._model.propagate_grad(
            g, h, moving, phi, left, right, inv)

        A = self.dat.affine[:-1, :-1].double()
        R = self.rotation[:-1, :-1].double()
        R = R[:, list(range(self.plane)) + list(range(self.plane+1, len(R)))]
        P = A.inverse() @ (R @ R.T) @ A
        mugrad = fastmath.matvec(P.to(mugrad), mugrad)

        return g, h, mugrad


class AffineModel(TransformationModel):
    """Affine transformation model encoded in a Lie algebra"""

    def __init__(self, basis, factor=1, penalty=None, dat=None, position='symmetric'):
        """

        Parameters
        ----------
        basis : {'translation', 'rotation', 'rigid', 'similitude', 'affine'}
            Affine basis name
        factor : float, default=1
            Regularization factor
        penalty : float or list[float] or dict(), optional
            L2 penalty
            If a list, one value per parameters.
            If a dict, keys "translation", "rotation", "zoom", "shear"
        dat : LogAffine or tensor, optional
            Pre-allocated log-parameters.
        position : {'moving', 'fixed', 'symmetric'}, default='symmetric'
            Which image should be rotated by this transformation.
            If 'symmetric', both images are rotated by the transformation and
            its inverse, towards a mean space; thereby making the model fully
            symmetric.
        """
        self.factor = factor
        self._penalty = penalty or None  # regutils.defaults_affine()
        self._penalty_list = None
        self._basis = basis
        self.position = position
        if dat is not None:
            self.set_dat(dat)
        else:
            self.dat = None

    @property
    def _truedim(self):
        return self.dat.dim if self.dat else None

    @property
    def penalty(self):
        if not self.dat or self._penalty is None:
            return None
        if self._penalty_list is None:
            B = self._basis[0].lower()
            D = self._truedim
            nb_t = D
            nb_r = (D*(D-1))//2
            nb_z0 = 1
            nb_z1 = D
            nb_z = nb_z0 if B == 's' else nb_z1
            nb_s = (D*(D-1))//2
            nb_p = nb_t
            if B != 't':
                nb_p += nb_r
            if B not in 'tr':
                nb_p += nb_z
            if B not in 'trs':
                nb_p += nb_s
            backend = dict(dtype=self.dat.dtype, device=self.dat.device)
            penalty = torch.zeros([nb_p], **backend)
            if isinstance(self._penalty_list, dict):
                for key, val in self._penalty_list.items():
                    key = key.lower()[0]
                    if key == 't':
                        penalty[:nb_t] = val
                    elif key == 'r':
                        penalty[nb_t:nb_t+nb_r] = val
                    elif key == 's':
                        penalty[nb_t+nb_r:nb_t+nb_r+nb_z] = val
                    elif key == 'a':
                        penalty[nb_t+nb_r+nb_z:] = val
            else:
                penalty0 = torch.as_tensor(self._penalty, **backend)
                B0 = ('a' if len(penalty0) == nb_t+nb_r+nb_z1+nb_s else
                      's' if len(penalty0) == nb_t+nb_r+nb_z0 else
                      'r' if len(penalty0) == nb_t+nb_r else
                      't' if len(penalty0) == nb_t else
                      '')
                if not B0 or B == B0:
                    penalty[...] = penalty0
                elif B in 'tr':
                    N = min(len(penalty), len(penalty0))
                    penalty[:N] = penalty0[:N]
                else:
                    N = min(len(penalty), len(penalty0), nb_t+nb_r)
                    penalty[:N] = penalty0[:N]
                    if B == 's' and B0 == 'a':
                        penalty[nb_t+nb_r] = penalty[nb_t+nb_r:nb_t+nb_r+nb_z1].mean()
                    elif B == 'a' and B0 == 's':
                        penalty[nb_t+nb_r:nb_t+nb_r+nb_z1] = penalty0[nb_t+nb_r]
            self._penalty_list = penalty
        return self._penalty_list

    def switch_basis(self, basis):
        obj = copy.deepcopy(self)
        if obj.dat:
            obj.dat = obj.dat.switch_basis(basis)
        obj._basis = basis
        obj._penalty_list = None
        return obj

    def set_dat(self, dat=None, dim=None, **backend):
        self.dat = LogAffine(dat, basis=self._basis, dim=dim, **backend)
        return self

    def parameters(self):
        if not self.dat:
            return None
        return self.dat.dat

    basis = property(lambda self: self.dat.basis if self.dat is not None else None)
    basis_name = property(lambda self: self._basis)

    def to(self, *args, **kwargs):
        return copy.copy(self).to_(*args, **kwargs)

    def to_(self, *args, **kwargs):
        if self.dat is not None:
            self.dat.to_(*args, **kwargs)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def cuda_(self):
        return self.to_('cuda')

    def cpu_(self):
        return self.to_('cpu')

    def clear_cache(self):
        if self.dat is not None:
            self.dat.clear_cache()

    def exp(self, q=None, grad=False, cache_result=False, recompute=True):
        return self.dat.exp(q, grad, cache_result, recompute)

    def iexp(self, q=None, grad=False, cache_result=False, recompute=True):
        return self.dat.iexp(q, grad, cache_result, recompute)

    def exp2(self, q=None, grad=False, cache_result=False, recompute=True):
        return self.dat.exp2(q, grad, cache_result, recompute)

    def __repr__(self):
        s = []
        if self.dat is not None:
            s += [f'log={self.dat}']
        else:
            s += ['<uninitialized>']
        s += [f'basis={self._basis}']
        s += [f'factor={self.factor}']
        s += [f'position={self.position}']
        if self.penalty is not None:
            s += [f'penalty={self.penalty.tolist()}']
        s = ', '.join(s)
        return f'{self.__class__.__name__}({s})'


class Affine2dModel(AffineModel):
    """A 2D affine in a 3D world"""

    def __init__(self, basis, plane, ref_affine=None, factor=1, penalty=None,
                 dat=None, position='symmetric'):
        super().__init__(basis, factor, penalty, dat, position)
        self._plane = plane
        self.ref_affine = ref_affine
        if ref_affine is None:
            rot = None
        else:
            rot = ref_affine.clone().double()
            rot[:-1, -1] = 0
            rot[:-1, :-1] /= rot[:-1, :-1].square().sum(0, keepdim=True).sqrt()
        self._rotation = rot.to(ref_affine)

    rotation = property(lambda self: self.dat.rotation if self.dat is not None else None)
    plane = property(lambda self: self.dat.plane if self.dat is not None else None)

    @property
    def _truedim(self):
        return 2

    @property
    def plane_name(self):
        return self._plane

    def set_dat(self, dat=None, dim=None, **backend):
        self.dat = LogAffine2d(dat, basis=self._basis, dim=dim,
                               rotation=self._rotation,
                               plane=self._plane, **backend)
        return self

