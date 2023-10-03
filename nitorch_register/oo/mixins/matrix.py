import torch
from torchrelay.multivers import linalg
from .shaped import MixinShaped


class MixinMatrix(MixinShaped):
    implements_matrix = True

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
        if not getattr(other, 'implements_vector', False):
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
        if not getattr(other, 'implements_matrix', False):
            raise ValueError('matmul only implemented for (Matrix, Matrix)')
        self = self.expand(other, part=('batch', 'spatial'))
        other = other.expand(other, part=('batch', 'spatial'))
        dat = self.dat.matmul(other.dat.unsqueeze(-1)).squeeze(-1)
        shapes = self.shapes.copy()
        shapes.item = [self.shapes.item[0], other.shapes.item[1]]
        return self.buildfrom(
            dat, cls=self.VectorType, attrs=dict(shapes=shapes)
        )
