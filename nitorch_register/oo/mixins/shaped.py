import torch
import numbers
from ..shapes import Shapes


class MixinShaped:
    implements_shaped = True

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
            if hasattr(shape, 'implements_shaped'):
                shape = shape.shapes
            elif hasattr(shape, 'shape'):
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
