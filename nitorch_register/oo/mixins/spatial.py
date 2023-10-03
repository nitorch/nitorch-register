from torchrelay.extra.bounds import all_bounds
from .shaped import MixinShaped


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
