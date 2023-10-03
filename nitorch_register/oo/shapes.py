import torch
import copy
import numbers


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

    def _to_str(self):
        comp = ', '.join([f'{key}={list(val)}' for key, val in self.items()])
        return f'Shapes({comp})'

    def __str__(self):
        return self._to_str()

    def __repr__(self):
        return self._to_str()

    def get(self, key, *args) -> torch.Size:
        """
        Get the size corresponding to a given key.
        If not present, return `Size([])`.

        ```
        >>> shape = Shapes(batch=[1, 1], item=[2, 2, 2])
        >>> shape.get('batch')
        torch.Size([1, 1])
        ```
        """
        if not isinstance(getattr(self, key, torch.Size([])), torch.Size):
            raise KeyError(f'Key {key} is protected')
        if args:
            return torch.Size(super().get(key, *args))
        else:
            return super().get(key, torch.Size([]))

    def __getitem__(self, key):
        """
        If `key` is an integer or slice, index into the full shape.
        If `key` is a string, return the subcomponent.

        ```
        >>> shape = Shapes(batch=[1, 1], item=[2, 2, 2])
        >>> shape['batch']
        torch.Size([1, 1])
        >>> shape[0]
        1
        ```
        """
        if isinstance(key, (numbers.Integral, slice)):
            return self.cat()[key]
        return self.get(key)

    def __setitem__(self, key, value):
        """
        If `key` is an integer or slice, assign into the full shape.
        If `key` is a string, assign a subcomponent.

        ```
        >>> shape = Shapes(batch=[1, 1], item=[2, 2, 2])

        >>> shape['batch'] = [1]
        >>> shape
        Shapes(batch=[1], item=[2, 2, 2])

        >>> shape[0] = 3
        >>> shape
        Shapes(batch=[3], item=[2, 2, 2])

        >>> shape[:2] = [5, 6]
        >>> shape
        Shapes(batch=[5], item=[6, 2, 2])
        ```
        """
        if isinstance(key, (numbers.Integral, slice)):
            index = key
            if isinstance(index, numbers.Integral):
                key = self.names()[key]
                index = len(self[key]) + index if index < 0 else index
                index = index - self.offsets()[key]
                self[key][index] = value
            else:
                index = list(range(len(self.cat())))[index]
                keys = self.names()[key]
                offsets = self.offsets()
                if isinstance(value, numbers.Integral):
                    value = [value] * len(index)
                for k, i, v in zip(keys, index, value):
                    self[k][i - offsets[k]] = v
            return
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
        (i.e., values -- instead of keys in a normal ictionary)
        """
        for val in self.values():
            yield val

    def permute(self, perm, key=None, other_left=True):
        """
        Reorder the subcomponents, or the dimensions of a subcomponent

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
        """
        Reorder the subcomponents, or the dimensions of a subcomponent

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
        """
        Alias for `permute(perm, key=None)`
        """
        return self.copy().reorder_(perm, other_left=other_left)

    def reorder_(self, perm, other_left=True):
        """
        Alias for `permute_(perm, key=None)`
        """
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

    def movedim(self, src: int, dst: int, key: str) -> "Shapes":
        """
        Move a dimension within a subcomponent
        """
        return self.copy().movedim_(src, dst, key)

    def movedim_(self, src: int, dst: int, key: str) -> "Shapes":
        """
        Move a dimension within a subcomponent
        """
        shape = list(self[key])
        src = shape.pop(src)
        shape.insert(dst, src)
        self[key] = shape
        return self

    def unsqueeze(self, dim: int, key: str) -> "Shapes":
        """
        Insert a singleton dimension
        """
        return self.copy().unsqueeze_(dim, key)

    def unsqueeze_(self, dim: int, key: str) -> "Shapes":
        """
        Insert a singleton dimension
        """
        shape = list(self[key])
        shape.insert(dim, 1)
        self[key] = shape

    def squeeze(self, dim: int = None, key: str = None) -> "Shapes":
        """
        Insert a singleton dimension
        """
        return self.copy().squeeze(dim, key)

    def squeeze_(self, dim: int = None, key: str = None) -> "Shapes":
        """
        Remove a singleton dimension.

        - If `key=None`, `dim` indexes into the full shape.
        - Else, `dim` indexes into the corresponding subcomponent(s).
        - If `dim=None`, all singleton dimensions are removed.

        Parameters
        ----------
        dim : int, optional
            Index of the dimension to remove
        key : str or sequence[str], optional
            Subsomponent(s) to squeeze.
        """
        if key is None and dim is None:
            # squeeze all singletons everywhere
            for key, val in self.items():
                self[key] = [s for s in val if s != 1]
        elif key is None:
            # squeeze dimension in full shape
            key = self.names()[dim]
            dim = len(self[key]) + dim if dim < 0 else dim
            offset = 0
            for k, v in self.items():
                if k == key:
                    break
                offset += len(v)
            shape = list(self[key])
            shape.pop(offset + dim)
            self[key] = shape
        else:
            # squeeze dimension in subcomponents
            keys = key
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                shape = list(self[key])
                shape.pop(dim)
                self[key] = shape
        return self

    def transpose(self, dim1, dim2, key=None) -> "Shapes":
        """
        Transpose two subcomponents, or two dimensions in a subcomponent.

        - If `key=None`, transpose two subcomponents
          (dim1 and dim2 are strings)
        - Else, transpose two dimensions in a subcomponent
          (dim1 and dim2 are integers)
        """
        return self.copy().transpose_(dim1, dim2, key)

    def transpose_(self, dim1, dim2, key=None) -> "Shapes":
        """
        Transpose two subcomponents, or two dimensions in a subcomponent.

        - If `key=None`, transpose two subcomponents
          (dim1 and dim2 are strings)
        - Else, transpose two dimensions in a subcomponent
          (dim1 and dim2 are integers)
        """
        if key is None:
            if not isinstance(dim1, str) or not isinstance(dim2, str):
                raise TypeError( "dimensions must be strings when `key=None`")
            keys = list(self.keys())
            keys[dim1], keys[dim2] = keys[dim2], keys[dim1]
            return self.reorder_(keys)
        else:
            dims = list(range(len(self[key])))
            dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
            return self.permute_(dims, key)

    def expand_(self, *shapes, key=None, exclude_key=None):
        """
        Expand / Broadcast

        If `part=None`, `shapes` must be `Shapes` (or objects that have a
        `shapes`, i.e., that have the attribute `implements_shaped = True`).
        Each subcomponents gets broadcasted.

        Else, `shapes` must be integer sequences (or object that have a shape),
        and the specific sucomponent gets broadcasted.

        Parameters
        ----------
        shapes : sequence[int] or Sizes or torch.Tensor or WrappedTensor
            Shapes (or shaped objects) to broadcast to
        key : str or list[str] or None
            Keys to expand. If None, expand all keys.
        exclude_key : str or list[str] or None
            Keys to not expand.
        """
        keys, exclude_keys = key, exclude_key
        if isinstance(keys, str):
            keys = [keys]
        if keys is not None:
            keys = list(keys)
        if isinstance(exclude_keys, str):
            exclude_keys = [exclude_keys]
        if exclude_keys is not None:
            exclude_keys = list(exclude_keys)
        else:
            exclude_keys = []
        if keys and set(keys).intersection(set(exclude_keys)):
            raise ValueError('Contradictory key is included _and_ excluded:',
                             set(keys).intersection(set(exclude_keys)))

        shape_dict = {}
        for shape in shapes:
            if getattr(shape, 'implements_shaped', False):
                shape = shape.shapes
            elif hasattr(shape, 'shape'):
                shape = shape.shape

            if isinstance(shape, Shapes):
                for k, v in shape.items():
                    if (k is None or k in keys) and k not in exclude_keys:
                        shape_dict.setdefault(k, [])
                        shape_dict[k].append(v)
            else:
                shape_dict.setdefault('item', [])
                shape_dict['item'].append(shape)

        for key, val in shape_dict.items():
            shapes[key] = torch.broadcast_shapes(val)



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

    def offsets():
        offsets = {}
        current_offset = 0
        for key, val in self.items():
            offsets[key] = current_offset
            current_offset += len(val)
        return offsets

    def numel(self):
        """
        Number of elements in a tensor of this size
        """
        return self.cat().numel()

    def copy(self):
        return copy.deepcopy(self)
