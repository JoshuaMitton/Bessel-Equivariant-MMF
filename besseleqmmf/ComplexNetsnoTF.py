import numpy as np
import cv2
import scipy.ndimage.filters as scifilters
from skimage.util import view_as_windows
import torch

def channels_to_complex(X):
    assert X.shape[-1] == 2
    return torch.view_as_complex(X)

def channels_to_complex_np(X):
    return X[..., 0] + 1j * X[..., 1]

def complex_to_channels(Z):
    return torch.view_as_real(Z)

def complex_to_channels_np(Z):
    RE = np.real(Z)
    IM = np.imag(Z)

    if Z.shape[-1] == 1:
        RE = np.squeeze(RE, (-1))
        IM = np.squeeze(IM, (-1))

    return np.stack([RE, IM], axis=-1)

def real_to_channels_np(X):
    import math
    # Create complex with zero imaginary part
    X_c = X + 0.j
    return complex_to_channels_np(X_c)

def rect_to_polar_np(X):
    Z = channels_to_complex_np(X)
    R = np.abs(Z)
    THETA = np.angle(Z)

    if Z.shape[-1] == 1:
        R = np.squeeze(R, (-1))
        THETA = np.squeeze(THETA, (-1))

    return np.stack([R, THETA], axis=-1)

def polar_to_rect_np(X):
    return complex_to_channels_np(X[..., 0] * np.exp(1j * X[..., 1]))

def real_to_channels_prop_np(r, max_val, max_phase_delay):
    theta = -max_phase_delay*np.pi*r/max_val
    polar = np.stack([r, theta], axis=-1)
    rect = polar_to_rect_np(polar)
    return rect

class Amplitude(torch.nn.Module):

    def __init__(self, **kwargs):
        super(Amplitude, self).__init__(**kwargs)

    def forward(self, X):
        complex_X = (channels_to_complex(X) if X.shape[-1] == 2 else X)
        output = torch.abs(complex_X)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

# Complex Dense Layer
class ComplexDense(torch.nn.Module):

    def __init__(self, in_features, out_features,
                 bias=True,
                 **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, 2))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features, 2))
        else:
            self.register_parameter('bias', None)
            
        torch.nn.init.uniform(self.weight,-np.sqrt(1/in_features),np.sqrt(1/in_features))
        torch.nn.init.uniform(self.bias,-np.sqrt(1/in_features),np.sqrt(1/in_features))
        

    def forward(self, X):
        # True Complex Multiplication (by channel combination)
        if X.dtype == torch.complex64:
            complex_X = X
        else:
            complex_X = channels_to_complex(X)
            complex_X = torch.flatten(complex_X, start_dim=-2)
        complex_W = channels_to_complex(self.weight)

        complex_res = complex_X @ torch.transpose(complex_W,0,1)
        
        if self.bias is not None:
            complex_b = channels_to_complex(self.bias)
            complex_res += complex_b
        
        output = complex_to_channels(complex_res)

        return output


