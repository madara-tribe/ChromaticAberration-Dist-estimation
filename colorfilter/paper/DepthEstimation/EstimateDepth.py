"""
Function Definitions:
estimate_depth(): Main function to estimate depth.
lhk(): Calculates the log-likelihood for different kernels.
lnerr(): Computes the Rank-1 approximation error.
pad_image(): Pads the image.
psf2otf(): Converts PSF to OTF.
kalign(): Calculates blur kernels.
getk(): Generates kernels.
candidate_depths(): Returns candidate radii.
bkernels(): Placeholder for generating blur kernels.
Image Processing:
Uses numpy, scipy, and opencv for image and matrix operations, Fourier transforms, and kernel manipulations.
Placeholder Functions:
Replace bkernels() with your actual kernel generation function.
Make sure to adjust paths and implement specific kernel generation functions as needed for your use case.
"""

import numpy as np
import cv2
from scipy.ndimage import convolve
from scipy.fft import fft2, ifft2, fftshift
from scipy.special import gamma

def estimate_depth(img):
    # Config parameters
    NSZ = 15  # Window size
    gsm = 1   # Derivative Scale
    krnls = kalign(10**-3, 5)  # Kernels
    blueAlign = 0  # Alignment flag

    # Construct gradient filters
    x = np.arange(-round(3*gsm), round(3*gsm) + 1)
    gss = 1/np.sqrt(2*np.pi)/gsm * np.exp(-x**2 / (2 * gsm**2))
    gsx = - (x / gsm**2) * gss
    gf = np.outer(gss, gsx) / np.sqrt(np.sum(np.square(np.outer(gss, gsx))))

    numl = len(krnls)  # Number of Candidate depths

    # Create "dummy" kernel
    sz = (len(krnls[0][0]) - 1) // 2
    dummy_kernel = cv2.getDerivKernels(1, 1, ksize=sz*2 + 1)[0][0]  # Disk filter
    krnls.append((dummy_kernel, dummy_kernel))

    # Compute likelihoods
    lhs = lhk(img, krnls, NSZ, gf, blueAlign)
    lhs = lhs[..., :numl] / np.abs(lhs[..., -1:])

    # Estimate depthmap
    dmap = np.zeros((img.shape[0], img.shape[1]), dtype=int)
    scr = -np.inf * np.ones((img.shape[0], img.shape[1]))

    for i in range(numl):
        lh = lhs[:, :, i]
        mask = lh > scr
        dmap[mask] = i
        scr[mask] = lh[mask]

    return dmap, lhs

def lhk(img, krnls, nsz, gf, blueAlign):
    flts = [np.ones((1, nsz)), np.ones((nsz, 1)), np.diag(np.ones(nsz)), np.fliplr(np.diag(np.ones(nsz)))]
    grds = [gf, gf.T, (gf + gf.T) / np.sqrt(2), (gf + gf.T) / np.sqrt(2)]

    imgder = [convolve(img[:, :, 0], grd, mode='constant') for grd in grds]
    
    psz = len(krnls[-1][0]) - 1
    im2f = fft2(pad_image(img[:, :, 1], psz))
    im3f = fft2(pad_image(img[:, :, 2], psz)) if blueAlign == 1 else None

    lh = np.zeros((img.shape[0], img.shape[1], len(krnls)))

    for i, (kr, kg) in enumerate(krnls):
        print(f'Testing for kernel {i+1} of {len(krnls)}')
        img2 = np.copy(img)

        if i > 0:
            tmp = np.real(ifft2(im2f * psf2otf(kr, im2f.shape)))
            img2[:, :, 1] = tmp[:img.shape[0], :img.shape[1]]
            if blueAlign == 1:
                tmp = np.real(ifft2(im3f * psf2otf(kg, im3f.shape)))
                img2[:, :, 2] = tmp[:img.shape[0], :img.shape[1]]

        for j, (flt, grd) in enumerate(zip(flts, grds)):
            dr = np.stack([convolve(img2[:, :, 0], grd, mode='constant'),
                           convolve(img2[:, :, 1], grd, mode='constant'),
                           convolve(img2[:, :, 2], grd, mode='constant')], axis=-1)

            v = np.copy(img2)
            for c in range(3):
                v[:, :, c] = convolve(img2[:, :, c], flt / np.sum(flt), mode='constant')

            R2 = convolve(dr[:, :, 0]**2, flt, mode='constant')
            G2 = convolve(dr[:, :, 1]**2, flt, mode='constant')
            B2 = convolve(dr[:, :, 2]**2, flt, mode='constant')
            RG = convolve(dr[:, :, 0] * dr[:, :, 1], flt, mode='constant')
            RB = convolve(dr[:, :, 0] * dr[:, :, 2], flt, mode='constant')
            GB = convolve(dr[:, :, 1] * dr[:, :, 2], flt, mode='constant')

            lh[:, :, i] -= lnerr(R2, G2, B2, RG, RB, GB, v)

    print()
    return lh

def lnerr(R2, G2, B2, RG, RB, GB, v2):
    maxiters = 3
    for _ in range(maxiters):
        s = np.sqrt(np.sum(v2**2, axis=-1))
        s[s == 0] = 1
        v = v2 / np.expand_dims(s, axis=-1)
        v2 = np.copy(v)
        v2[:, :, 0] = v[:, :, 0] * R2 + v[:, :, 1] * RG + v[:, :, 2] * RB
        v2[:, :, 1] = v[:, :, 0] * RG + v[:, :, 1] * G2 + v[:, :, 2] * GB
        v2[:, :, 2] = v[:, :, 0] * RB + v[:, :, 1] * GB + v[:, :, 2] * B2

    egv = np.sum(v2 * v, axis=-1)
    smv = R2 + G2 + B2
    return smv - egv

def pad_image(img, psz):
    psz = psz // 2
    img = np.hstack([img, np.tile(img[:, -1:], (1, psz))])
    img = np.vstack([img, np.tile(img[-1:, :], (psz, 1))])
    return img

def psf2otf(psf, shape):
    psf = np.pad(psf, [(0, shape[0] - psf.shape[0]), (0, shape[1] - psf.shape[1])], mode='constant')
    return fft2(psf)

def kalign(mu, fctr):
    depths = [0] + candidate_depths()
    krnls = []
    sz = round(max(depths) * fctr) * 2 + 1

    for d in depths:
        if d > 0:
            kr, kg, kb = bkernels(d)
            kr = kr / np.sum(kr)
            kg = kg / np.sum(kg)
            kb = kb / np.sum(kb)
        else:
            kr = np.ones((sz, sz))
            kg = np.ones((sz, sz))
            kb = np.ones((sz, sz))
        krnls.append((getk(kr, kg, sz, mu), getk(kr, kb, sz, mu)))

    return krnls

def getk(k1, k2, sz, mu):
    sz = (sz, sz)
    den1 = np.abs(psf2otf([-1, 1], sz))**2 / 2 + np.abs(psf2otf([-1, 1], sz))**2 / 2
    k1f = psf2otf(k1, sz)
    k2f = psf2otf(k2, sz)
    kf = k1f * np.conj(k2f) / (mu * den1 + np.abs(k2f)**2)
    return fftshift(np.real(ifft2(kf)))

def candidate_depths():
    return 2 * np.arange(1, 13)

def bkernels(depth):
    # Placeholder implementation - replace with actual kernel generation
    size = 2 * depth + 1
    kr = np.ones((size, size))
    kg = np.ones((size, size))
    kb = np.ones((size, size))
    return kr, kg, kb

# Example usage
img = cv2.imread('image.png')  # Replace with the path to your image
dmap, lhs = estimate_depth(img)


