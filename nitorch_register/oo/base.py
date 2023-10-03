import torch
import numbers
import copy
import nitorch_interpol as interpol
from torchrelay.multivers import linalg
from torchrelay.extra.bounds import all_bounds
from .backends import is_numpy, from_numpy, is_cupy, from_cupy


class WrappedTensor:

    def __init__(self, tensor, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        tensor : torch.tensor or np.ndarray or cp.ndarray
            Input tensor

        Other Parameters
        ----------------
        dtype : torch.dtype
            Data type. By default, same as input.
        device : torch.device
            Device. By default, same as input.
        memory_format : torch.memory_format
            The desired memory format of returned Tensor.
            Default: `torch.preserve_format`
        copy : bool
            When `copy` is set, a new Tensor is created even when the
            tensor already matches the desired conversion.
        non_blocking : bool
            When `non_blocking`, tries to convert asynchronously with
            respect to the host if possible, e.g., converting a CPU
            tensor with pinned memory to a CUDA tensor.
        attrs : dict
            Dictionary or attributes to set
        """
        self.initfrom(tensor, *args, **kwargs)

    # ------------------------------------------------------------------
    #       Attributes
    # ------------------------------------------------------------------

    def getattrs(self):
        return {}

    def setattrs_(self, **attr):
        for key, value in attr.items():
            setattr(self, key, value)
        return self

    def copyattrs_(self, other):
        return self.setattrs(**other.getattrs())

    def delattrs_(self, *args):
        for key in args:
            delattr(self, key)

    @property
    def shape(self):
        return self.dat.shape

    @property
    def dtype(self):
        return self.dat.dtype

    @property
    def device(self):
        return self.dat.device

    @property
    def requires_grad(self):
        return self.dat.requires_grad

    @property
    def requires_grad_(self, value=True):
        self.dat.requires_grad_(value)
        return self

    # ------------------------------------------------------------------
    #       Constructors
    # ------------------------------------------------------------------

    @classmethod
    def _preptensor(cls, tensor, *args, **kwargs):
        """Convert tensor to correct backend/dtype/device"""
        if is_numpy(tensor):
            tensor = from_numpy(tensor)
        if is_cupy(tensor):
            tensor = from_cupy(tensor)
        return tensor.to(*args, **kwargs)

    def initfrom(self, other, *args, **kwargs):
        """Initialize object's state from (wrapped) tensor

        Parameters
        ----------
        other : torch.Tensor or WrappedTensor
            Tensor to initialize from

        Other Parameters
        ----------------
        dtype : torch.dtype
            Data type. By default, same as input.
        device : torch.device
            Device. By default, same as input.
        memory_format : torch.memory_format
            The desired memory format of returned Tensor.
            Default: `torch.preserve_format`
        copy : bool
            When `copy` is set, a new Tensor is created even when the
            tensor already matches the desired conversion.
        non_blocking : bool
            When `non_blocking`, tries to convert asynchronously with
            respect to the host if possible, e.g., converting a CPU
            tensor with pinned memory to a CUDA tensor.
        attrs : dict
            Dictionary or attributes to set

        """
        attrs = kwargs.pop(attrs, {})
        if isinstance(other, WrappedTensor):
            for key, value in other.getattrs().items():
                attrs.setdefaut(key, value)
            other = other.dat
        self.dat = self._preptensor(other, *args, **kwargs)
        self.setattrs_(attrs)
        return self

    @classmethod
    def buildfrom(cls, other, *args, **kwargs):
        """Build object from (wrapped) tensor

        Parameters
        ----------
        other : torch.Tensor or WrappedTensor
            Tensor to initialize from

        Other Parameters
        ----------------
        attrs : dict
            Dictionary or attributes to set
        dtype : torch.dtype
            Data type. By default, same as input.
        device : torch.device
            Device. By default, same as input.
        memory_format : torch.memory_format
            The desired memory format of returned Tensor.
            Default: `torch.preserve_format`
        copy : bool
            When `copy` is set, a new Tensor is created even when the
            tensor already matches the desired conversion.
        non_blocking : bool
            When `non_blocking`, tries to convert asynchronously with
            respect to the host if possible, e.g., converting a CPU
            tensor with pinned memory to a CUDA tensor.

        """
        cls = kwargs.pop('cls', cls)
        return cls(other, *args, **kwargs)

    def buildfrom(self, other, *args, **kwargs):
        """Build object from (wrapped) tensor, and copy current attributes

        Parameters
        ----------
        other : torch.Tensor or WrappedTensor
            Tensor to initialize from

        Other Parameters
        ----------------
        attrs : dict
            Dictionary or attributes to set
        dtype : torch.dtype
            Data type. By default, same as input.
        device : torch.device
            Device. By default, same as input.
        memory_format : torch.memory_format
            The desired memory format of returned Tensor.
            Default: `torch.preserve_format`
        copy : bool
            When `copy` is set, a new Tensor is created even when the
            tensor already matches the desired conversion.
        non_blocking : bool
            When `non_blocking`, tries to convert asynchronously with
            respect to the host if possible, e.g., converting a CPU
            tensor with pinned memory to a CUDA tensor.

        """
        attrs = kwargs.pop('attrs', {})
        attrs.update(self.getattrs())
        cls = kwargs.pop('cls', type(self))
        return cls.buildfrom(other, *args, **kwargs, attrs=attrs)

    @classmethod
    def _build(cls, func, shape, *args, **kwargs):
        attrs = kwargs.pop('attrs', {})
        cls = kwargs.pop('cls', cls)
        return cls.buildfrom(func(shape, *args, **kwargs), attrs=attrs)

    @classmethod
    def empty(cls, shape, *args, **kwargs):
        return cls._build(torch.empty, shape, *args, **kwargs)

    @classmethod
    def zeros(cls, shape, *args, **kwargs):
        return cls._build(torch.zeros, shape, *args, **kwargs)

    @classmethod
    def ones(cls, shape, *args, **kwargs):
        return cls._build(torch.ones, shape, *args, **kwargs)

    @classmethod
    def full(cls, shape, *args, **kwargs):
        return cls._build(torch.full, shape, *args, **kwargs)

    @classmethod
    def rand(cls, shape, *args, **kwargs):
        return cls._build(torch.rand, shape, *args, **kwargs)

    @classmethod
    def randn(cls, shape, *args, **kwargs):
        return cls._build(torch.randn, shape, *args, **kwargs)

    @classmethod
    def _build_like(cls, func, other, *args, **kwargs):
        attrs = kwargs.pop('attrs', {})
        if isinstance(other, WrappedTensor):
            for key, value in other.getattrs().items():
                attrs.setdefaut(key, value)
            other = other.dat
        if torch.is_tensor(other):
            return cls(func(other, *args, **kwargs), attrs=attrs)
        if hasattr(other, 'shape'):
            kwargs.setdefault('dtype', getattr(other, 'dtype', None))
            kwargs.setdefault('device', getattr(other, 'device', None))
            return cls(func(other.shape, *args, **kwargs), attrs=attrs)
        raise TypeError(
            f"I don't know what to do with object of type {type(other)}"
        )

    @classmethod
    def empty_like(cls, other, *args, **kwargs):
        return cls._build_like(torch.empty_like, other, *args, **kwargs)

    @classmethod
    def zeros_like(cls, other, *args, **kwargs):
        return cls._build_like(torch.zeros_like, other, *args, **kwargs)

    @classmethod
    def ones_like(cls, other, *args, **kwargs):
        return cls._build_like(torch.ones_like, other, *args, **kwargs)

    @classmethod
    def full_like(cls, other, *args, **kwargs):
        return cls._build_like(torch.full_like, other, *args, **kwargs)

    @classmethod
    def rand_like(cls, other, *args, **kwargs):
        return cls._build_like(torch.rand_like, other, *args, **kwargs)

    @classmethod
    def randn_like(cls, other, *args, **kwargs):
        return cls._build_like(torch.randn_like, other, *args, **kwargs)

    # ------------------------------------------------------------------
    #       Accessors
    # ------------------------------------------------------------------

    def slice(self, slicer, **kwargs):
        return self.buildfrom(self.dat[slicer], **kwargs)

    def slice_put(self, slicer, other, **kwargs):
        self.dat[slicer] = other
        return self.slice(slicer, **kwargs)

    def __getitem__(self, slicer):
        return self.slice(slicer)

    def __setitem__(self, slicer, value):
        return self.slice_put(slicer, value)

    # ------------------------------------------------------------------
    #       Shape / view
    # ------------------------------------------------------------------

    def reshape(self, shape):
        """Reshape the tensor

        Parameters
        ----------
        shape : sequence[int]
            New shape. The total number of elements must be consistent
            with the previous shape.

        Returns
        -------
        type(self)
        """
        return self.buildfrom(self.dat.reshape(shape))

    def unsqueeze(self, dim):
        """Add a singleton dimension

        Parameters
        ----------
        dim : int
            Index of the dimension to insert.

        Returns
        -------
        type(self)
        """
        return self.buildfrom(self.dat.unsqueeze(dim))

    def squeeze(self, dim=None):
        """Remove a singleton dimension

        Parameters
        ----------
        dim : int, optional
            Index of the singleton dimension to remove.
            If `None`, remove all singleton dimensions.

        Returns
        -------
        type(self)
        """
        return self.buildfrom(self.dat.squeeze(dim))

    def expand(self, *shapes):
        """Expand/broadcast the shape of a tensor

        Parameters
        ----------
        shapes : torch.Tensor or WrappedTensor or sequence[int]
            Shapes to broadcast

        Returns
        -------
        type(self)
        """
        shapes = [getattr(x, 'shape', x) for x in shapes]
        return self.buildfrom(self.dat.expand(*shapes))

    def permute(self, perm):
        """Permute the dimensions of the tensor

        Parameters
        ----------
        perm : sequence[int]
            Permutation

        Returns
        -------
        type(self)
        """
        return self.buildfrom(self.dat.permute(perm))

    def movedim(self, src, dst):
        """Move a dimension of the tensor

        Parameters
        ----------
        src : int
            Source dimensions
        dst : int
            Destination dimensions

        Returns
        -------
        type(self)
        """
        return self.buildfrom(self.dat.movedim(src, dst))

    def transpose(self, dim1, dim2):
        """Transpose/exchange two dimensions

        Parameters
        ----------
        dim1 : int
            Index of the fist dimension
        dim2 : int
            Index of the second dimension

        Returns
        -------
        type(self)
        """
        return self.buildfrom(self.dat.transpose(dim1, dim2))


    # ------------------------------------------------------------------
    #       Math op : utilties
    # ------------------------------------------------------------------

    def _math1(self, func, *args, **kwargs):
        attrs = kwargs.pop('attrs', {})
        cls = kwargs.pop('cls', type(self))
        return self.buildfrom(
            func(self.dat, *args, **kwargs), cls=cls, attrs=attrs
        )

    def _math1_inplace(self, func_, *args, **kwargs):
        func_(self.dat, *args, **kwargs)
        return self

    def _math2(self, func, other, *args, **kwargs):
        attrs = kwargs.pop('attrs', {})
        cls = kwargs.pop('cls', type(self))
        if isinstance(other, WrappedTensor):
            other = other.dat
        elif not torch.is_tensor(other) or isinstance(other, numbers.Number):
            raise TypeError(
                f"I don't know what to do with object of type {type(other)}"
            )
        return self.buildfrom(
            func(self.dat, other, *args, **kwargs), cls=cls, attrs=attrs
        )

    def _math2_inplace(self, func_, other, *args, **kwargs):
        if isinstance(other, WrappedTensor):
            other = other.dat
        elif not torch.is_tensor(other) or isinstance(other, numbers.Number):
            raise TypeError(
                f"I don't know what to do with object of type {type(other)}"
            )
        func_(self.dat, other, *args, **kwargs)
        return self


class Shapes(dict):
    """A container for shapes with multiple components"""

    default_order = ('batch', 'space', 'time', 'item')

    def __init__(self, *args, **kwargs):
        """
        Build from a dictionary or a set of keys
        """
        # make sure we have the correct order for standard keys
        for key in self.default_order:
            self[key] = tuple()
        super().__init__(*args, **kwargs)

    def get(self, key, *args) -> torch.Size:
        """
        Get the size corresponding to a given key.
        If not present, return `Size([])`.
        """
        if not isinstance(getattr(self, key, torch.Size([])), torch.Size):
            raise KeyError(f'Key {key} is protected')
        if args:
            return torch.Size(super().get(key, *args))
        else:
            return super().get(key, torch.Size([]))

    def __getitem__(self, key) -> torch.Size:
        return self.get(key)

    def __setitem__(self, key, value):
        if not isinstance(getattr(self, key, torch.Size([])), torch.Size):
            raise KeyError(f'Key {key} is protected')
        super().__setitem__(key, torch.Size(value))

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        if isinstance(getattr(self, name), torch.Size):
            self[name] = torch.Size(value)
        else:
            super().__setattr__(name, value)

    def __iter__(self):
        """
        Iterate over subcomponents
        (i.e., values -- instead of keys in a normal dictionary)
        """
        for val in self.values():
            yield val

    def permute(self, perm, key=None, other_left=True):
        """Reorder the subcomponents, or the dimensions of a subcomponent

        If `key=None`, `perm` should contain strings (component names)
        ```
        self.permute(['item', 'batch'])
        ```

        If `key` is a string, `perm` should contain integers (dim indices)
        ```
        self.permute([0, 2, 1], key='item')
        ```
        """
        return self.copy().permute_(perm, key=key, other_left=other_left)

    def permute_(self, perm, key=None, other_left=True):
        """Reorder the subcomponents, or the dimensions of a subcomponent

        If `key=None`, `perm` should contain strings (component names)
        ```
        self.permute_(['item', 'batch'])
        ```

        If `key` is a string, `perm` should contain integers (dim indices)
        ```
        self.permute_([0, 2, 1], key='item')
        ```
        """
        if key is None:
            return self.reorder_(perm, other_left=other_left)
        else:
            new_shape = [self[key][p] for p in perm]
            other = [i for i in range(len(self[key])) if i not in perm]
            if other_left:
                new_shape = other + new_shape
            else:
                new_shape = new_shape + other
            self[key] = new_shape
        return self

    def reorder(self, perm, other_left=True):
        return self.copy().reorder_(perm, other_left=other_left)

    def reorder_(self, perm, other_left=True):
        """Alias for `permute(perm, key=None)`"""
        tmp = dict(self)
        for key in self.keys():
            del self[key]
        # put keys that are absent from `keys` to the left
        if other_left:
            for key in tmp.keys():
                if key not in perm:
                    self[key] = tmp[key]
        # assign keys that are in `keys` in order
        for key in perm:
            self[key] = tmp[key]
        # put keys that are absent from `keys` to the right
        if not other_left:
            for key in tmp.keys():
                if key not in perm:
                    self[key] = tmp[key]
        return self

    def cat(self) -> torch.Size:
        """
        Return the full, concatenated shape
        """
        shape = torch.Size([])
        for part in self:
            shape = shape + part
        return shape

    def split_(self, fullshape):
        """
        Assign subcomponents from a full concatenated shape.

        Requires that `len(self.cat()) == len(fullshape)`
        """
        fullshape = torch.Size(fullshape)
        if len(fullshape) != len(self.cat()):
            raise ValueError('Cannot split a shape with different number '
                             'of dimensions than current shape')
        offset = 0
        for key in self.keys():
            length = len(self[key])
            self[key] = fullshape[offset:offset+length]
            offset += length
        return self

    def names(self):
        """
        Return the name of the subcomponent corresponding to each full index

        ```
        Shapes(batch=(1, 1), item=(2, 2, 2))
        >>> ['batch', 'batch', 'item', 'item', 'item']
        ```
        """
        names = []
        for key, val in self.items():
            names += [key] * len(val)
        return names

    def numel(self):
        return self.cat().numel()

    def copy(self):
        return copy.deepcopy(self)


class MixinShaped:

    # ------------------------------------------------------------------
    #       Shape / view
    # ------------------------------------------------------------------

    def reshape(self, shape=None, **kwargs):
        """Reshape the tensor

        Parameters
        ----------
        shape : Shapes, optional
            Shape of each component

        Other Parameters
        ----------------
        batch : sequence[int]
            Batch shape
        space : sequence[int]
            Spatial shape
        time : sequence[int]
            Time shape
        item : sequence[int]
            Item shape
        ...

        Returns
        -------
        reshaped_tensor : WrappedTensor
        """
        if shape is not None:
            shapes = shape.copy()
            for key, val in kwargs.items():
                shapes.setdefault(key, val)
            for key, val in self.shapes.items():
                shapes.setdefault(key, val)
        else:
            shapes = Shapes(**kwargs)
        return self.buildfrom(
            self.dat.reshape(shape), attrs=dict(shapes=shapes)
        )

    def unsqueeze(self, dim, part):
        """Add a singleton dimension

        Parameters
        ----------
        dim : int
            Index of dimension to unsqueeze (within the part)
        part : {'batch', 'space', 'time', 'item', ...}
            Shape part to unsqueeze

        Returns
        -------
        unsequeezed_tensor : WrappedTensor
        """
        shapes = self.shapes.copy()
        old_shape = getattr(shapes, part)
        new_shape = torch.empty([]).expand(old_shape).unsqueeze(dim).shape
        setattr(shapes, part, new_shape)
        return self.reshape(shapes)

    def squeeze(self, dim=None, part=None):
        """Remove a singleton dimension

        Parameters
        ----------
        dim : int
            Index of dimension to squeeze.
            If `part` is None, `dim` indexes into the full `shape` vector.
            Else, `dim` indexes into the specific part's `shape` vector.
            If `dim` is None, squeeze all singleton dimensions.
        part : {None, 'batch', 'space', 'time', 'item', ...}
            Shape part to squeeze

        Returns
        -------
        squeezed_tensor : WrappedTensor
        """
        shapes = self.shapes.copy()
        if part is None and dim is None:
            for key, val in shapes.items():
                shapes[key] = torch.empty([]).expand(val).squeeze().shape
        elif part is None:
            parts = shapes.names()
            part = parts[dim]
            fullshape = list(shapes.cat())
            del fullshape[dim]
            shapes[key] = shapes[key][:-1]
            shapes = shapes.split_(fullshape)
        else:
            old_shape = getattr(shapes, part)
            new_shape = torch.empty([]).expand(old_shape).squeeze().shape
            shapes[part] = new_shape
        return self.reshape(shapes)

    def expand(self, *shapes, part=None):
        """Expand/broadcast the shape of a tensor

        Parameters
        ----------
        shapes : torch.Tensor or WrappedTensor or Shape or sequence[int]
            Shapes to broadcast
        part : None or [list of] {'batch', 'space', 'time', 'item', ...}
            Shape part to expand.
            `None` expands all possible parts.

        Returns
        -------
        expanded_tensor : WrappedTensor
        """
        if isinstance(part, numbers.Integral):
            part = [part]
        if part is not None:
            part = list(part)
        shape_dict = {}
        for shape in shapes:
            if torch.is_tensor(shape):
                shape = shape.shape
            if isinstance(shape, WrappedTensor):
                if hasattr(shape, 'shapes'):
                    shape = shape.shapes
                else:
                    shape = shape.shape
            if isinstance(shape, Shapes):
                for key, val in shape.items():
                    if part is None or key in part:
                        shape_dict.setdefault(key, [])
                        shape_dict[key].append(val)
            elif part is None or 'item' in part:
                shape_dict.setdefault('item', [])
                shape_dict['item'].append(shape)
        shapes = self.shapes.copy()
        for key, val in shape_dict.items():
            shapes[key] = torch.broadcast_shapes(val)
        return self.reshape(shapes)

    def permute(self, perm, part):
        """Permute the dimensions of the tensor

        Parameters
        ----------
        perm : sequence[int]
            Permutation
        part : {'batch', 'space', 'time', 'item', ...}
            Shape part to shuffle

        Returns
        -------
        expanded_tensor : WrappedTensor
        """
        shapes = self.shapes.copy()
        old_shape = getattr(shapes, part)
        perm = [len(old_shape) + p if p < 0 else p for p in perm]

        offset = 0
        for key, val in shapes:
            if key == part:
                break
            else:
                offset += len(val)
        perm = list(range(offset)) + [p+offset for p in perm]
        perm = perm + list(range(len(perm), len(shapes.cat())))

        dat = self.dat.permute(perm)
        shapes.split_(dat.shape)
        return self.buildfrom(dat, attr=dict(shapes=shapes))

    def movedim(self, src, dst, part):
        """Move a dimension

        Parameters
        ----------
        src : int
            Source dimensions
        dst : int
            Destination dimensions
        part : {'batch', 'space', 'time', 'item', ...}
            Shape part to shuffle

        Returns
        -------
        permuted_tensor : WrappedTensor
        """
        shapes = self.shapes.copy()
        old_shape = getattr(shapes, part)
        src = len(old_shape) + src if src < 0 else src
        dst = len(old_shape) + dst if dst < 0 else dst
        pre, moved, post = old_shape[:src], old_shape[src], old_shape[src+1:]
        new_shape = pre + post
        new_shape = new_shape[:dst] + (moved,) + new_shape[dst:]
        setattr(shapes, part, new_shape)
        shapes.split_(shapes)
        dat = self.dat.movedim(src, dst)
        return self.buildfrom(dat, attr=dict(shapes=shapes))

    def transpose(self, dim1, dim2, part):
        """Transpose/exchange two dimensions

        Parameters
        ----------
        dim1 : int
            First dimension
        dim2 : int
            Second dimensions
        part : {'batch', 'space', 'item'}
            Shape part to shuffle

        Returns
        -------
        permuted_tensor : WrappedTensor
        """
        old_shape = getattr(self.shapes, part)
        perm = list(range(len(old_shape)))
        dim1 = len(old_shape) + dim1 if dim1 < 0 else dim1
        dim2 = len(old_shape) + dim2 if dim2 < 0 else dim2
        perm[dim1], perm[dim2] = dim2, dim1
        return self.permute(perm, part)


class MixinSpatial(MixinShaped):

    # ------------------------------------------------------------------
    #       attributes
    # ------------------------------------------------------------------

    @property
    def bound(self):
        return getattr(self, '_bound', 'zero')

    @bound.setter
    def bound(self, value):
        if value not in all_bounds:
            raise ValueError('Unknown bound', value)
        self._bound = value

    @property
    def extrapolate(self):
        return getattr(self, '_extrapolate', True)

    @extrapolate.setter
    def extrapolate(self, value):
        if not isinstance(value, bool):
            raise TypeError('Extrapolation mode should be a boolean')
        self._extrapolate = value

    @property
    def spline_order(self):
        return getattr(self, '_spline_order', 1)

    @spline_order.setter
    def spline_order(self, value):
        if not isinstance(value, int) or not (0 <= value <= 7):
            raise TypeError('Spline order should be an integer in 0..7')
        self._spline_order = value

    # ------------------------------------------------------------------
    #       finite differences
    # ------------------------------------------------------------------

    def difference(dim=None, part='space', unit='voxel', mode='central'):
        pass

    def divergence(dim=None, part='space', unit='voxel', mode='central'):
        pass

    # ------------------------------------------------------------------
    #       interpolation / sampling
    # ------------------------------------------------------------------

    def pull(phi, **kwargs):
        pass

    def push(phi, out=None, **kwargs):
        pass

    def pushcount(phi, out=None, **kwargs):
        pass

    def pullgrad(phi, unit='voxel', space='native', **kwargs):
        pass


class MixinSpatialMasked(MixinSpatial):

    # ------------------------------------------------------------------
    #       interpolation / sampling
    # ------------------------------------------------------------------

    def pull(phi, **kwargs):
        pass

    def push(phi, out=None, **kwargs):
        pass

    def pushcount(phi, out=None, **kwargs):
        pass

    def pullgrad(phi, unit='voxel', space='native', **kwargs):
        pass


class MixinElementwise:

    # ------------------------------------------------------------------
    #       one argument
    # ------------------------------------------------------------------

    def neg(self, *args, **kwargs):
        return self._math1(torch.neg, *args, **kwargs)

    def neg_(self, *args, **kwargs):
        return self._math1_inplace(torch.neg_, *args, **kwargs)

    def negative(self, *args, **kwargs):
        return self._math1(torch.negative, *args, **kwargs)

    def negative_(self, *args, **kwargs):
        return self._math1_inplace(torch.negative_, *args, **kwargs)

    def reciprocal(self, *args, **kwargs):
        return self._math1(torch.reciprocal, *args, **kwargs)

    def reciprocal_(self, *args, **kwargs):
        return self._math1_inplace(torch.reciprocal_, *args, **kwargs)

    def abs(self, *args, **kwargs):
        return self._math1(torch.abs, *args, **kwargs)

    def abs_(self, *args, **kwargs):
        return self._math1_inplace(torch.abs_, *args, **kwargs)

    def angle(self, *args, **kwargs):
        return self._math1(torch.angle, *args, **kwargs)

    def angle_(self, *args, **kwargs):
        return self._math1_inplace(torch.angle_, *args, **kwargs)

    # ------------------------------------------------------------------
    #       two arguments
    # ------------------------------------------------------------------

    def add(self, other, *args, **kwargs):
        self = self.expand(other)
        return self._math2(torch.add, other, *args, **kwargs)

    def add_(self, other, *args, **kwargs):
        self = self.expand(other)
        return self._math2_inplace(torch.Tensor.add_, other, *args, **kwargs)

    def sub(self, other, *args, **kwargs):
        self = self.expand(other)
        return self._math2(torch.sub, other, *args, **kwargs)

    def sub_(self, other, *args, **kwargs):
        self = self.expand(other)
        return self._math2_inplace(torch.Tensor.sub_, other, *args, **kwargs)


class MixinVector(MixinShaped):

    # ------------------------------------------------------------------
    #       one argument
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #       two arguments
    # ------------------------------------------------------------------

    def vecmat(self, other, *args, **kwargs):
        if not isinstance(other, MixinMatrix):
            raise ValueError('vecmat only implemented for (Vector, Matrix)')
        self = self.expand(other, part=('batch', 'spatial'))
        other = other.expand(other, part=('batch', 'spatial'))
        dat = self.dat.unsqueeze(-2).matmul(other.dat).squeeze(-2)
        shapes = self.shapes.copy()
        shapes.item = other.shape[1:]
        return self.buildfrom(dat, attrs=dict(shapes=shapes))

    def cross(self, other, *args, **kwargs):
        if not isinstance(other, MixinVector):
            raise ValueError('cross only implemented for (Vector, Vector)')
        if self.shape[-1] != 3 or other.shape[-1] != 3:
            raise ValueError('cross only implemented for 3D vectors')
        self = self.expand(other)
        return self._math2(linalg.cross, other, *args, **kwargs)


class MixinMatrix(MixinShaped):

    # ------------------------------------------------------------------
    #       one argument
    # ------------------------------------------------------------------

    def det(self, *args, **kwargs):
        shapes = self.shapes.copy()
        shapes.item = tuple()
        return self._math1(
            linalg.det, attrs=dict(shapes=shapes), cls=self.ScalarType
        )

    def logdet(self, *args, **kwargs):
        shapes = self.shapes.copy()
        shapes.item = tuple()
        return self.buildfrom(
            linalg.logdet, attrs=dict(shapes=shapes), cls=self.ScalarType
        )

    def trace(self, *args, **kwargs):
        shapes = self.shapes.copy()
        shapes.item = tuple()
        return self.buildfrom(
            linalg.trace, attrs=dict(shapes=shapes), cls=self.VectorType
        )

    def transpose(self, dim1=-2, dim2=-1, part='item'):
        return super().transpose(dim1, dim2, part)

    def adjoint(self, dim1=-2, dim2=-1, part='item'):
        return super().transpose(dim1, dim2, part).conj()

    @property
    def T(self):
        return self.transpose()

    @property
    def H(self):
        return self.adjoint()

    # ------------------------------------------------------------------
    #       two arguments
    # ------------------------------------------------------------------

    def matvec(self, other, *args, **kwargs):
        if not isinstance(other, MixinVector):
            raise ValueError('matvec only implemented for (Matrix, Vector)')
        self = self.expand(other, part=('batch', 'spatial'))
        other = other.expand(other, part=('batch', 'spatial'))
        dat = self.dat.matmul(other.dat.unsqueeze(-1)).squeeze(-1)
        shapes = self.shapes.copy()
        shapes.item = self.shapes.item[:1]
        return self.buildfrom(
            dat, cls=self.VectorType, attrs=dict(shapes=shapes)
        )

    def matmul(self, other, *args, **kwargs):
        if not isinstance(other, MixinMatrix):
            raise ValueError('matmul only implemented for (Matrix, Matrix)')
        self = self.expand(other, part=('batch', 'spatial'))
        other = other.expand(other, part=('batch', 'spatial'))
        dat = self.dat.matmul(other.dat.unsqueeze(-1)).squeeze(-1)
        shapes = self.shapes.copy()
        shapes.item = [self.shapes.item[0], other.shapes.item[1]]
        return self.buildfrom(
            dat, cls=self.VectorType, attrs=dict(shapes=shapes)
        )


class MixinTrigo:

    # ------------------------------------------------------------------
    #       one argument
    # ------------------------------------------------------------------

    def acos(self, *args, **kwargs):
        return self._math1(torch.acos, *args, **kwargs)

    def acos_(self, *args, **kwargs):
        return self._math1_inplace(torch.acos_, *args, **kwargs)

    def acosh(self, *args, **kwargs):
        return self._math1(torch.acosh, *args, **kwargs)

    def acosh_(self, *args, **kwargs):
        return self._math1_inplace(torch.acosh_, *args, **kwargs)

    def asin(self, *args, **kwargs):
        return self._math1(torch.asin, *args, **kwargs)

    def asin_(self, *args, **kwargs):
        return self._math1_inplace(torch.asin_, *args, **kwargs)

    def asinh(self, *args, **kwargs):
        return self._math1(torch.asinh, *args, **kwargs)

    def asinh_(self, *args, **kwargs):
        return self._math1_inplace(torch.asinh_, *args, **kwargs)

    def atan(self, *args, **kwargs):
        return self._math1(torch.atan, *args, **kwargs)

    def atan_(self, *args, **kwargs):
        return self._math1_inplace(torch.atan_, *args, **kwargs)

    def atanh(self, *args, **kwargs):
        return self._math1(torch.atanh, *args, **kwargs)

    def atanh_(self, *args, **kwargs):
        return self._math1_inplace(torch.atanh_, *args, **kwargs)

    # ------------------------------------------------------------------
    #       two arguments
    # ------------------------------------------------------------------

    def atan2(self, *args, **kwargs):
        return self._math2(torch.atan2, *args, **kwargs)

    def atan2_(self, *args, **kwargs):
        return self._math2_inplace(torch.Tensor.atan2_, *args, **kwargs)
