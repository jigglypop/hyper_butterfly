import torch

try:
    from . import _rust
except ImportError:
    # A trick to make the linter happy, but this will raise a proper ImportError
    # if the rust backend is not built
    _rust = object()
    raise

class Manifold:
    def __init__(self, c=1.0):
        self.c = c
class Poincare(Manifold):
    def mobius_add(self, x, y):
        return _rust.mobius_add(x, y, self.c)
    def mobius_scalar(self, x, r):
        return _rust.mobius_scalar(x, r, self.c)
    def distance(self, x, y):
        return _rust.poincare_distance(x, y, self.c)
    def poincare_layer(self, start, target, t):
        scaled_start = self.mobius_scalar(start, 1.0 - t)
        scaled_target = self.mobius_scalar(target, t)
        return self.mobius_add(scaled_start, scaled_target)
    def to_lorentz(self, x):
        return _rust.poincare_to_lorentz(x, self.c)
    def to_klein(self, x):
        return _rust.poincare_to_klein(x, self.c)
class Lorentz(Manifold):
    def add(self, x, y):
        return _rust.lorentz_add(x, y, self.c)
    def scalar(self, x, r):
        return _rust.lorentz_scalar(x, r, self.c)
    def distance(self, x, y):
        return _rust.lorentz_distance(x, y, self.c)
    def inner(self, x, y):
        return _rust.lorentz_inner(x, y)
    def to_poincare(self, x):
        return _rust.lorentz_to_poincare(x, self.c)
    def to_klein(self, x):
        return _rust.lorentz_to_klein(x, self.c)
class Klein(Manifold):
    def add(self, x, y):
        return _rust.klein_add(x, y, self.c)
    def scalar(self, x, r):
        return _rust.klein_scalar(x, r, self.c)
    def distance(self, x, y):
        return _rust.klein_distance(x, y, self.c)
    def to_poincare(self, x):
        return _rust.klein_to_poincare(x, self.c)
    def to_lorentz(self, x):
        return _rust.klein_to_lorentz(x, self.c) 