import torch
from torchrelay.multivers import linalg
from .shaped import MixinShaped


class MixinVector(MixinShaped):
    implements_vector = True

    # ------------------------------------------------------------------
    #       one argument
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #       two arguments
    # ------------------------------------------------------------------

    def vecmat(self, other, *args, **kwargs):
        if not getattr(other, 'implements_matrix', False):
            raise ValueError('vecmat only implemented for (Vector, Matrix)')
        self = self.expand(other, part=('batch', 'spatial'))
        other = other.expand(other, part=('batch', 'spatial'))
        dat = self.dat.unsqueeze(-2).matmul(other.dat).squeeze(-2)
        shapes = self.shapes.copy()
        shapes.item = other.shape[1:]
        return self.buildfrom(dat, attrs=dict(shapes=shapes))

    def cross(self, other, *args, **kwargs):
        if not getattr(other, 'implements_vector', False):
            raise ValueError('cross only implemented for (Vector, Vector)')
        if self.shape[-1] != 3 or other.shape[-1] != 3:
            raise ValueError('cross only implemented for 3D vectors')
        self = self.expand(other)
        return self._math2(linalg.cross, other, *args, **kwargs)
