"""
Explanation:
ftv Function:
Calls fpc to prepare precomputed parameters.
Calls ftvq to perform the deconvolution with the precomputed parameters and regularization weight.
fpc Function:
Computes the FFT of the blur kernels and other precomputed parameters.
ftvq Function:
Uses precomputed parameters to perform iterative deconvolution.
padimg Function:
Pads the image to handle boundary effects during convolution.
precomp Function:
Computes values used in the iterative process of deconvolution.
gets Function:
Determines the shrinkage threshold and scaling parameters for gradient shrinkage.
shrinkv23 Function:
A placeholder for the actual shrinkage function. You should replace this with your specific implementation or use an existing shrinkage function.
Make sure to test and validate the translated code with your specific data to ensure it works as expected.
"""

import numpy as np
from scipy.fft import fft2, ifft2
import cv2

def ftv(y, k1, k2, k3, mu):
    """
    Fast Deconvolution with color constraints on gradients
    
    Parameters:
    y: Three channel blurred image
    k1, k2, k3: Blur kernels acting on Channels 1,2,3
    mu: Regularization weight
    
    Returns:
    x: Deconvolved Image
    """
    pc = fpc(k1, k2, k3, y.shape)
    x = ftvq(y, pc, mu)
    return x

def fpc(k1, k2, k3, image_size):
    """
    Compute precomputed parameters for deconvolution.
    
    Parameters:
    k1, k2, k3: Blur kernels for each color channel
    image_size: Size of the input image (height, width)
    
    Returns:
    pc: A list of precomputed parameters including the kernels and their FFTs
    """
    # Compute the kernels in frequency domain
    k1f = fft2(k1, s=image_size)
    k2f = fft2(k2, s=image_size)
    k3f = fft2(k3, s=image_size)

    # Compute the precomputed parameters
    den1 = np.abs(k1f)**2 + np.abs(k2f)**2 + np.abs(k3f)**2
    den2 = np.conj(k1f) * k1f + np.conj(k2f) * k2f + np.conj(k3f) * k3f
    
    # Pack the parameters into a list
    pc = [0, k1f, k2f, k3f, den1, den2]
    
    return pc

def ftvq(y, pc, mu):
    """
    Fast Deconvolution with pre-computed blur parameters
    
    Parameters:
    y: Three channel blurred image
    pc: Output of fpc, which has been called with acting blur kernels
    mu: Regularization weight
    
    Returns:
    x: Deconvolved Image
    """
    out_it = 8
    beta_s = 4
    beta = (100 * mu) / (beta_s ** out_it)

    # Padding the image
    pad = pc[0]
    y_padded = padimg(y, pad)

    # Precompute necessary quantities
    num2, den1, den2 = precomp(y_padded, pc)

    # Initialization
    x = np.copy(y_padded)
    fct = 0.5
    for i in range(3):
        x[..., i] = np.real(ifft2(fct * num2[..., i] / (den1 + fct * den2[..., i])))

    # Find gradient directions
    w0x, w0y = np.gradient(x, axis=(1, 0))
    w0n = np.sqrt(np.sum(w0x ** 2, axis=2, keepdims=True))
    w0x = np.where(w0n == 0, 1 / np.sqrt(3), w0x / w0n)
    w0n = np.sqrt(np.sum(w0y ** 2, axis=2, keepdims=True))
    w0y = np.where(w0n == 0, 1 / np.sqrt(3), w0y / w0n)

    for _ in range(out_it):
        # W sub-problem
        wx = np.diff(x, axis=1) / np.sqrt(2)
        wy = np.diff(x, axis=0) / np.sqrt(2)

        # Project
        wxj = np.sum(wx * w0x, axis=2)
        wyj = np.sum(wy * w0y, axis=2)

        # Shrinkage
        mx = max(1e-4, np.max(np.abs(wxj)), np.max(np.abs(wyj)))
        thr, a, b = gets(beta, mx)

        wxj = np.where(np.abs(wxj) < thr, 0, np.maximum(0, a * np.abs(wxj) + b) * np.sign(wxj))
        wyj = np.where(np.abs(wyj) < thr, 0, np.maximum(0, a * np.abs(wyj) + b) * np.sign(wyj))

        # Final gradient vectors
        wx = wxj[..., np.newaxis] * w0x
        wy = wyj[..., np.newaxis] * w0y

        # X sub-problem
        for i in range(3):
            num1 = np.concatenate([
                -wx[:, 0, i:i+1], -np.diff(wx[..., i:i+1], axis=1), wx[:, -1, i:i+1]], axis=1
            ) + np.concatenate([
                -wy[0, :, i:i+1], -np.diff(wy[..., i:i+1], axis=0), wy[-1, :, i:i+1]], axis=0
            )
            tmp = fft2(num1 / np.sqrt(2)) + mu / beta * num2[..., i]
            tmp = tmp / (den1 + mu / beta * den2[..., i])
            x[..., i] = np.real(ifft2(tmp))

        beta *= beta_s  # Beta continuation

    # Remove padding
    x = x[pad:-pad, pad:-pad, :]
    return x

def padimg(y, pad):
    """
    Pad the image with a border
    
    Parameters:
    y: Input image
    pad: Padding size
    
    Returns:
    y2: Padded image
    """
    y2 = np.zeros((y.shape[0] + 2 * pad, y.shape[1] + 2 * pad, y.shape[2]))
    wt = np.linspace(0, 1, 2 * pad)
    wtx = np.tile(wt[np.newaxis, :], (y.shape[0], 1))
    wty = np.tile(wt[:, np.newaxis], (1, y2.shape[1]))
    for i in range(3):
        yi = y[..., i]
        px = (1 - wtx) * yi[:, -1][:, np.newaxis] + wtx * yi[:, 0][:, np.newaxis]
        yi = np.hstack([px[:, pad:], yi, px[:, :pad]])
        py = (1 - wty) * yi[-1, :] + wty * yi[0, :]
        yi = np.vstack([py[pad:, :], yi, py[:pad, :]])
        y2[..., i] = yi
    return y2

def precomp(y, pc):
    """
    Precompute values used in the deconvolution
    
    Parameters:
    y: Padded image
    pc: Precomputed parameters
    
    Returns:
    num2, den1, den2: Precomputed values
    """
    den1 = pc[5]
    den2 = pc[6]
    k1f, k2f, k3f = pc[1], pc[2], pc[3]

    num2 = np.stack([np.conj(k1f) * fft2(y[..., 0]),
                     np.conj(k2f) * fft2(y[..., 1]),
                     np.conj(k3f) * fft2(y[..., 2])], axis=-1)
    return num2, den1, den2

def gets(beta, mx):
    """
    Compute shrinkage parameters for gradient thresholding
    
    Parameters:
    beta: Regularization parameter
    mx: Maximum gradient value
    
    Returns:
    thr, a, b: Threshold and scaling parameters
    """
    v = np.linspace(0, mx, 5000)
    v2 = shrinkv23(v, beta)
    idx = np.min(np.where(v2 > 0))

    if np.isnan(idx):
        return 0, 0, 10

    thr = v[idx]
    a = (v2[-1] - v2[idx]) / (v[-1] - v[idx])
    b = v2[-1] - a * v[-1]
    return thr, a, b

def shrinkv23(v, beta):
    """
    Compute shrinkage function values
    
    Parameters:
    v: Gradient magnitudes
    beta: Regularization parameter
    
    Returns:
    w: Shrinkage values
    """
    # Placeholder for the actual implementation
    # This is where you should implement or replace with your shrinkage function
    # For now, this is just a placeholder
    w = np.maximum(0, v - beta)
    return w

