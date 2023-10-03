import torch


class MixinElementwise:
    implements_math = True

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


class MixinTrigo:
    implements_trigo = True

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
