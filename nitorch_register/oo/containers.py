import torch
from .base import (
    WrappedTensor,
    Shapes,
    MixinVector,
    MixinMatrix,
    MixinShaped,
    MixinTrigo,
    MixinElementwise,
)


class ShapedTensor(
    MixinShaped,
    WrappedTensor,
):
    pass


class Scalar(
    MixinTrigo,
    MixinElementwise,
    ShapedTensor
):

    # ------------------------------------------------------------------
    #       Attributes
    # ------------------------------------------------------------------

    @property
    def shapes(self):
        return Shapes(batch=self.shape)

    # ------------------------------------------------------------------
    #       Shape / view
    # ------------------------------------------------------------------

    def as_vector(self):
        return Vector(self.dat[..., None])

    def as_matrix(self):
        return Matrix(self.dat[..., None, None])

    def as_tensor3(self):
        return Tensor3(self.dat[..., None, None, None])

    def as_tensor(self, ndim):
        if ndim == 0:
            return self
        if ndim == 1:
            return self.as_vector()
        if ndim == 2:
            return self.as_matrix()
        if ndim == 3:
            return self.as_tensor3()
        dat = self.dat
        for _ in range(ndim):
            dat = dat.unsqueeze(-1)
        return Tensor(dat, ndim)

    def as_spatial(self, ndim):
        dat = self.dat
        for _ in range(ndim):
            dat = dat.unsqueeze(-1)
        return SpatialScalar(dat, ndim)


class Vector(
    MixinVector,
    MixinTrigo,
    MixinElementwise,
    ShapedTensor
):

    def __init__(self, input, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self.shapes = Shapes(input.shape[:-1], tuple(), input.shape[-1:])

    @property
    def vecsize(self):
        return self.shapes.item[0]

    # ------------------------------------------------------------------
    #       Shape / view
    # ------------------------------------------------------------------

    def as_colmat(self):
        return Matrix(self.dat[..., None])

    def as_rowmat(self):
        return Matrix(self.dat[..., None, :])

    def as_matrix(self):
        return self.as_colmat()

class Matrix(
    MixinMatrix,
    MixinTrigo,
    MixinElementwise,
    WrappedTensor
):

    def __init__(self, input, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self.shapes = Shapes(input.shape[:-2], tuple(), input.shape[-2:])

    # ------------------------------------------------------------------
    #       Shape / view
    # ------------------------------------------------------------------

    def as_colmat(self):
        return Matrix(self.dat[..., None])

    def as_rowmat(self):
        return Matrix(self.dat[..., None, :])

    def as_matrix(self):
        return self.as_colmat()


class Tensor3(
    MixinTrigo,
    MixinElementwise,
    ShapedTensor
):

    def __init__(self, input, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self.shapes = Shapes(input.shape[:-3], tuple(), input.shape[-3:])

    # ------------------------------------------------------------------
    #       Shape / view
    # ------------------------------------------------------------------

    def as_colmat(self):
        return Matrix(self.dat[..., None])

    def as_rowmat(self):
        return Matrix(self.dat[..., None, :])

    def as_matrix(self):
        return self.as_colmat()



ShapedTensor.ScalarType = Scalar
ShapedTensor.VectorType = Vector
ShapedTensor.MatrixType = Matrix
ShapedTensor.Tensor3Type = Tensor3
