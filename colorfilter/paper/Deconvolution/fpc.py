"""
Explanation:
psf2otf Function:
Converts the Point Spread Function (PSF) to the Optical Transfer Function (OTF) by padding the PSF and computing its FFT.
fpc Function:
Computes the necessary precomputed parameters for deconvolution.
Pads the image size based on the kernel size.
Computes den1 and den2 using the psf2otf function to get the Fourier Transforms of the kernels.
Returns a list of precomputed parameters.


You can use this fpc function in conjunction with the previously provided ftv and ftvq functions. The fpc function prepares the necessary parameters that are used by ftvq to perform the deconvolution.

If you need further modifications or have any other specific requirements, let me know!
"""

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import fourier_shift

def psf2otf(psf, size):
    """
    Convert Point Spread Function (PSF) to Optical Transfer Function (OTF).
    
    Parameters:
    psf: Point Spread Function
    size: Size of the output (height, width)
    
    Returns:
    otf: Optical Transfer Function
    """
    psf_padded = np.zeros(size)
    psf_padded[:psf.shape[0], :psf.shape[1]] = psf
    otf = fft2(psf_padded)
    return otf

def fpc(k1, k2, k3, sz):
    """
    Compute precomputed parameters for deconvolution.
    
    Parameters:
    k1, k2, k3: Blur kernels for each color channel
    sz: Size of the input image (height, width)
    
    Returns:
    pc: A list of precomputed parameters including the kernels and their FFTs
    """
    pad = (len(k1) - 1) // 2 * 3

    sz = (sz[0] + 2 * pad, sz[1] + 2 * pad)

    # Compute den1
    den1 = np.abs(psf2otf([-1, 1], sz))**2 / 2 + np.abs(psf2otf([-1, 1], sz[::-1]))**2 / 2

    # Compute kernels in frequency domain
    k1f = psf2otf(k1, sz)
    k2f = psf2otf(k2, sz)
    k3f = psf2otf(k3, sz)

    # Compute den2
    den2 = np.zeros((sz[0], sz[1], 3), dtype=np.complex_)
    den2[..., 0] = np.abs(k1f)**2
    den2[..., 1] = np.abs(k2f)**2
    den2[..., 2] = np.abs(k3f)**2

    # Pack the parameters into a list
    pc = [pad, k1f, k2f, k3f, den1, den2]
    
    return pc

