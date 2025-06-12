"""
Reality Stone - High-performance hyperbolic neural networks
Powered by Rust + PyTorch
"""

__version__ = "0.2.0"

import torch
import numpy as np
from . import _rust

# Constants
EPS = 1e-7
BOUNDARY_EPS = 1e-5
MIN_DENOMINATOR = 1e-6

def _to_numpy(tensor):
    """Convert PyTorch tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def _from_numpy(array, device=None, dtype=torch.float32):
    """Convert numpy array to PyTorch tensor"""
    tensor = torch.from_numpy(array).to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

# Mobius operations
def mobius_add(u, v, c=1.0):
    """Möbius addition in the Poincaré ball"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    v_np = _to_numpy(v).astype(np.float32)
    result = _rust.mobius_add(u_np, v_np, float(c))
    return _from_numpy(result, device)

def mobius_scalar(u, r, c=1.0):
    """Möbius scalar multiplication in the Poincaré ball"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    result = _rust.mobius_scalar(u_np, float(c), float(r))
    return _from_numpy(result, device)

# Poincaré operations
def poincare_distance(u, v, c=1.0):
    """Distance in the Poincaré ball"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    v_np = _to_numpy(v).astype(np.float32)
    result = _rust.poincare_distance(u_np, v_np, float(c))
    return _from_numpy(result, device)

def poincare_to_lorentz(x, c=1.0):
    """Convert from Poincaré ball to Lorentz model"""
    device = x.device
    x_np = _to_numpy(x).astype(np.float32)
    result = _rust.poincare_to_lorentz(x_np, float(c))
    return _from_numpy(result, device)

def poincare_to_klein(x, c=1.0):
    """Convert from Poincaré ball to Klein model"""
    device = x.device
    x_np = _to_numpy(x).astype(np.float32)
    result = _rust.poincare_to_klein(x_np, float(c))
    return _from_numpy(result, device)

# Lorentz operations
def lorentz_add(u, v, c=1.0):
    """Addition in the Lorentz model"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    v_np = _to_numpy(v).astype(np.float32)
    result = _rust.lorentz_add(u_np, v_np, float(c))
    return _from_numpy(result, device)

def lorentz_scalar(u, r, c=1.0):
    """Scalar multiplication in the Lorentz model"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    result = _rust.lorentz_scalar(u_np, float(c), float(r))
    return _from_numpy(result, device)

def lorentz_distance(u, v, c=1.0):
    """Distance in the Lorentz model"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    v_np = _to_numpy(v).astype(np.float32)
    result = _rust.lorentz_distance(u_np, v_np, float(c))
    return _from_numpy(result, device)

def lorentz_inner(u, v):
    """Minkowski inner product"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    v_np = _to_numpy(v).astype(np.float32)
    result = _rust.lorentz_inner(u_np, v_np)
    return _from_numpy(result, device)

def lorentz_to_poincare(x, c=1.0):
    """Convert from Lorentz model to Poincaré ball"""
    device = x.device
    x_np = _to_numpy(x).astype(np.float32)
    result = _rust.lorentz_to_poincare(x_np, float(c))
    return _from_numpy(result, device)

def lorentz_to_klein(x, c=1.0):
    """Convert from Lorentz model to Klein model"""
    device = x.device
    x_np = _to_numpy(x).astype(np.float32)
    result = _rust.lorentz_to_klein(x_np, float(c))
    return _from_numpy(result, device)

# Klein operations
def klein_add(u, v, c=1.0):
    """Addition in the Klein model"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    v_np = _to_numpy(v).astype(np.float32)
    result = _rust.klein_add(u_np, v_np, float(c))
    return _from_numpy(result, device)

def klein_scalar(u, r, c=1.0):
    """Scalar multiplication in the Klein model"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    result = _rust.klein_scalar(u_np, float(c), float(r))
    return _from_numpy(result, device)

def klein_distance(u, v, c=1.0):
    """Distance in the Klein model"""
    device = u.device
    u_np = _to_numpy(u).astype(np.float32)
    v_np = _to_numpy(v).astype(np.float32)
    result = _rust.klein_distance(u_np, v_np, float(c))
    return _from_numpy(result, device)

def klein_to_poincare(x, c=1.0):
    """Convert from Klein model to Poincaré ball"""
    device = x.device
    x_np = _to_numpy(x).astype(np.float32)
    result = _rust.klein_to_poincare(x_np, float(c))
    return _from_numpy(result, device)

def klein_to_lorentz(x, c=1.0):
    """Convert from Klein model to Lorentz model"""
    device = x.device
    x_np = _to_numpy(x).astype(np.float32)
    result = _rust.klein_to_lorentz(x_np, float(c))
    return _from_numpy(result, device)

# Convenience aliases
poincare_ball_add = mobius_add
poincare_ball_scalar = mobius_scalar

def poincare_ball_layer(start_point, target_point, c, t):
    minus_start = mobius_scalar(start_point, -1.0, c)
    delta = mobius_add(minus_start, target_point, c)
    delta_t = mobius_scalar(delta, t, c)
    return mobius_add(start_point, delta_t, c)

__all__ = [
    'mobius_add', 'mobius_scalar',
    'poincare_distance', 'poincare_to_lorentz', 'poincare_to_klein',
    'lorentz_add', 'lorentz_scalar', 'lorentz_distance', 'lorentz_inner',
    'lorentz_to_poincare', 'lorentz_to_klein',
    'klein_add', 'klein_scalar', 'klein_distance',
    'klein_to_poincare', 'klein_to_lorentz',
    'poincare_ball_add', 'poincare_ball_scalar',
    'poincare_ball_layer'
] 