#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:02:05 2021

@author: moli
"""



import struct
import numpy as np
from scipy import stats
    
from scipy import fftpack
# import radialProfile

# https://www.unioviedo.es/compnum/labs/PYTHON/lab06_Fourier2D.html
# from __future__ import division             # forces floating point division 
import numpy as np                          # Numerical Python 
from PIL import Image                       # Python Imaging Library
from numpy.fft import fft2, fftshift, ifft2 # Python DFT




def write_double_binary(fichier, data): #write_double_binary(file, data)
    try:
        with open(fichier, 'wb') as file:
            for elem in data:
                file.write(struct.pack('<d', elem))
    except IOError:
        print('Erreur ecriture.')



def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


iso_turb = np.fromfile('Initial_W_discret/file2D.bin', dtype='<f8', count=-1)
scale = int(np.sqrt(len(iso_turb)))
image = np.reshape(iso_turb, [scale, scale])
# Take the fourier transform of the image.
F1 = fftpack.fft2(image)
    
# Now shift the quadrants around so that low spatial frequencies are in
# the center of the 2D fourier transformed image.
F2 = fftpack.fftshift( F1 )
    
# Calculate a 2D power spectrum
psd2D = np.abs( F2 )**2
psd1D = azimuthalAverage(psd2D)
            
res = 1024*8
amp = int(res**2/512**2)
F2_2048 = np.zeros((res, res), dtype=complex)
F2_2048[int(res/2-256):int(res/2+256), int(res/2-256):int(res/2+256)] = F2
F1_2048 = fftpack.fftshift( F2_2048 )
image_2048 = np.zeros((res, res))
image_2048 = np.real(fftpack.ifft2(F1_2048))
    
write_double_binary('Initial_W_discret/file2D_'+str(res)+'.bin', image_2048.flatten()*amp)
    



    