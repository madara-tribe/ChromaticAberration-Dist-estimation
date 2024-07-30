"""
Explanation:
Alpha Ratio Calculation:
frat = 26 / 44 is a direct conversion from the MATLAB code.
Kernel Creation:
cv2.getDerivKernels(0, 0, ksize) generates derivative kernels (disk-shaped) in OpenCV. For a disk of radius depth, it needs to be adjusted by multiplying depth * 2 + 1 for the size.
k2a is similarly generated but scaled by frat.
Kernel Convolution:
convolve() from scipy.ndimage performs the convolution. It’s similar to MATLAB’s conv2().
Normalization:
Normalize k1 by dividing by its sum.
Normalize k2 by dividing by its maximum value and then scaling to the maximum value of k1.
Return Values:
Return the kernels kr, kg, and kb similar to the MATLAB function.
Note:

In this implementation, cv2.getDerivKernels is used for generating disk-like kernels, though in real scenarios, you might need to use other methods to simulate exact disk-shaped kernels if cv2.getDerivKernels doesn’t perfectly match fspecial('disk', ...) from MATLAB. The fspecial('disk', ...) is specific to MATLAB, and replicating exact disk-shaped filters might need custom implementation using scipy or numpy functions for a more accurate match.
"""

import numpy as np
import cv2
from scipy.ndimage import convolve

def bkernels(depth):
    # Alpha ratio
    frat = 26 / 44

    # Create the primary kernel using a disk filter
    k1 = cv2.getDerivKernels(0, 0, ksize=depth*2 + 1)[0][0]
    
    # Create the secondary kernel
    k2a = cv2.getDerivKernels(0, 0, ksize=int(depth * frat * 2) + 1)[0][0]

    # Initialize k2
    k2 = np.zeros_like(k1)
    k2[depth, depth] = 1
    
    # Convolve k2 with k2a
    k2 = convolve(k2, k2a, mode='constant', cval=0.0)
    
    # Normalize k1 and k2
    k1 /= np.sum(k1)
    k2 = k2 / np.max(k2) * np.max(k1)
    
    # Output kernels
    kr = k1
    kg = k2
    kb = k1

    return kr, kg, kb

# Example usage
depth = 5  # Example depth
kr, kg, kb = bkernels(depth)


