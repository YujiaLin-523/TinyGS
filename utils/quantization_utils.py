import torch
import numpy as np

class HalfSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.half().float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
def half_ste(input):
    return HalfSTEFunction.apply(input)

def log_transform(data):
    """Apply signed log(1+x) transform element-wise.

    Original implementation performs in-place masked assignment which fails
    for numpy scalars (0-d arrays) because they don't support item assignment.
    This version handles both scalars and ndarrays uniformly and always
    returns a numpy ndarray (0-d for scalar input) to keep downstream .max()/.min().
    """
    arr = np.asarray(data)
    # 0-d array (scalar) special case
    if arr.ndim == 0:
        val = float(arr)
        if val > 0:
            return np.asarray(np.log1p(val), dtype=np.float32)
        elif val < 0:
            return np.asarray(-np.log1p(-val), dtype=np.float32)
        else:
            return np.asarray(val, dtype=np.float32)
    # For regular arrays, operate on a copy to avoid side-effects
    out = arr.copy()
    positive = out > 0
    negative = out < 0
    out[positive] = np.log1p(out[positive])
    out[negative] = -np.log1p(-out[negative])
    return out.astype(np.float32)

def inverse_log_transform(data):
    """Inverse of log_transform supporting scalar and ndarray inputs."""
    arr = np.asarray(data)
    if arr.ndim == 0:
        val = float(arr)
        if val > 0:
            return np.asarray(np.expm1(val), dtype=np.float32)
        elif val < 0:
            return np.asarray(-np.expm1(-val), dtype=np.float32)
        else:
            return np.asarray(val, dtype=np.float32)
    out = arr.copy()
    positive = out > 0
    negative = out < 0
    out[positive] = np.expm1(out[positive])
    out[negative] = -np.expm1(-out[negative])
    return out.astype(np.float32)

def distributal_clip(data, bit=8):
    d = 3 + 3 * (bit - 1) / 15
    mean, std = data.mean(), data.std()
    return data.clip(mean - d * std, mean + d * std)

def quantize(data, bit=8, log=False, clip=True):
    """Uniform (or log) quantization supporting scalar inputs.

    Ensures data is an ndarray so .max()/.min() are available. Returns:
      data_q: uint8 ndarray
      data_scale: float scale factor
      data_min: original min before scaling
    """
    arr = np.asarray(data, dtype=np.float32)
    if clip:
        arr = distributal_clip(arr, bit)
    if log:
        arr = log_transform(arr)
    data_max, data_min = arr.max(), arr.min()
    # Prevent divide-by-zero if all values identical
    if data_max == data_min:
        data_scale = 1.0
        data_q = np.zeros_like(arr, dtype=np.uint8)
        return data_q, data_scale, data_min
    data_scale = (2**bit - 1) / (data_max - data_min)
    data_q = np.clip(np.round((arr - data_min) * data_scale), 0, 2**bit - 1).astype(np.uint8)
    return data_q, data_scale, data_min

def dequantize(data_quant, data_scale, data_min, log=False):
    data_q = (data_quant.astype(np.float32) / data_scale) + data_min
    if log:
        data_q = inverse_log_transform(data_q)

    return data_q
