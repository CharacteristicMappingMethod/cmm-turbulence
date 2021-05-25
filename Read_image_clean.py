#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:02:05 2021

@author: moli
"""



import struct
import matplotlib
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
    
from scipy import fftpack
import pylab as py
# import radialProfile

# https://www.unioviedo.es/compnum/labs/PYTHON/lab06_Fourier2D.html
# from __future__ import division             # forces floating point division 
import numpy as np                          # Numerical Python 
import matplotlib.pyplot as plt             # Python plotting
from PIL import Image                       # Python Imaging Library
from numpy.fft import fft2, fftshift, ifft2 # Python DFT
import farge_colormaps


import pywt
import pywt.data



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



plt.close('all')


OS = "Ubuntu"
# OS = "Windows"

if OS == "Ubuntu":
    chemin = "/home/moli/Bureau/Document_Ubuntu"
elif OS == "Windows":
    chemin = "D:/Travail/These/Document_Ubuntu"


    
def Fourier():
    
    
    # image = pywt.data.camera()
    # plt.figure()
    # plt.imshow(image, origin = 'upper',  cmap = plt.cm.gray )
    # # plt.imshow(image_2048*amp, origin = 'upper',  cmap = plt.cm.gray )
    
    
    iso_turb = np.fromfile(chemin+'/Code_cmm_cuda2d_Moli/src/Initial_W_discret/file2D.bin', dtype='<f8', count=-1)
    scale = int(np.sqrt(len(iso_turb)))
    image = np.reshape(iso_turb, [scale, scale])
    plt.figure()
    plt.imshow(image, origin = 'lower',  cmap ='jet', )
    
    
    
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(image)
    
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift( F1 )
    
    # Calculate a 2D power spectrum
    psd2D = np.abs( F2 )**2
    
    # Calculate the azimuthally averaged 1D power spectrum
    # psd1D = radialProfile.azimuthalAverage(psd2D)
    psd1D = azimuthalAverage(psd2D)
    
    # py.semilogy(psd1D)
    Spatial_Frequency = (np.arange(len(psd1D))+1)#*2/scale*(2*np.pi)
    plt.figure()
    plt.loglog(Spatial_Frequency, psd1D)
    py.xlabel('Spatial Frequency')
    py.ylabel('Power Spectrum')
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(Spatial_Frequency[:160]), np.log10(psd1D[:160]))
    print(slope)
    
    # plt.figure()
    # plt.imshow(np.log10(abs(F1)), origin = 'lower',  cmap ='jet', )
    plt.figure()
    plt.imshow(np.log10(psd2D), origin = 'lower',  cmap ='jet', )
    
    
    res = 2048*2
    amp = int(res**2/512**2)
    F2_2048 = np.zeros((res, res), dtype=complex)
    F2_2048[int(res/2-256):int(res/2+256), int(res/2-256):int(res/2+256)] = F2
    F1_2048 = fftpack.fftshift( F2_2048 )
    image_2048 = np.zeros((res, res))
    image_2048 = np.real(fftpack.ifft2(F1_2048))
    
    plt.figure()
    plt.imshow(np.log10(np.abs( F2_2048 )**2), origin = 'lower',  cmap ='jet', )
    # plt.figure()
    # plt.imshow(np.log10(np.abs( F1_2048 )**2), origin = 'lower',  cmap ='jet', )
    plt.figure()
    plt.imshow(image_2048*amp, origin = 'lower',  cmap ='jet', )
    
    # write_double_binary(chemin+'/Code_cmm_cuda2d_Moli/src/Initial_W_discret/file2D_'+str(res)+'.bin', image_2048.flatten()*amp)
    
    
    F1_real = np.real(F1)
    # F1_real = np.imag(F1)
    image_real = np.real(fftpack.ifft2(F1_real))
    # plt.imshow(np.log10(abs(F1_real)), origin = 'lower',  cmap ='jet', )
    
    plt.figure()
    plt.imshow(image_real, origin = 'lower',  cmap ='jet', )
    
    plt.figure()
    plt.imshow(image, origin = 'lower',  cmap ='jet', )
    
    imageimage = image+image[::-1,::-1]
    plt.figure()
    plt.imshow(imageimage, origin = 'lower',  cmap ='jet', )
    

    
def Turbulence():
    
    iso_turb = np.fromfile(chemin+'/Code_cmm_cuda2d_Moli/src/Initial_W_discret/file2D.bin', dtype='<f8', count=-1)
    scale = int(np.sqrt(len(iso_turb)))
    image = np.reshape(iso_turb, [scale, scale])
    plt.figure()
    plt.imshow(image, origin = 'lower',  cmap ='jet', )
    plt.imshow(fftpack.fftshift( image ), origin = 'lower',  cmap ='jet', )
    write_double_binary(chemin+'/Code_cmm_cuda2d_Moli/src/Initial_W_discret/file2D_512.bin', fftpack.fftshift( image ).flatten())
    
    
    _ = 'final'
    plt.figure()
    for _ in range(1, 100):
        print(_)
        iso_turb = np.fromfile(chemin+'/Code_cmm_cuda2d_Moli/data/vortex_shear_1000_4/all_save_data/w_'+str(_)+'.data', dtype='<f8', count=-1)
        scale = int(np.sqrt(len(iso_turb)))
        image = np.reshape(iso_turb, [scale, scale])
        plt.clf()
        plt.imshow(image, origin = 'lower',  cmap ='jet', )
        # plt.pcolormesh(image, cmap=farge_colormaps.farge_colormap_multi(type='vorticity'))
        plt.colorbar()
        plt.pause(0.1)



    plt.figure()
    for _ in range(1, 100, 5):
        print(_)
        
        iso_turb = np.fromfile(chemin+'/Code_cmm_cuda2d_Moli/data/vortex_shear_1000_4/all_save_data/w_'+str(_)+'.data', dtype='<f8', count=-1)
        scale = int(np.sqrt(len(iso_turb)))
        image = np.reshape(iso_turb, [scale, scale])
        
        # Take the fourier transform of the image.
        F1 = fftpack.fft2(image)
        
        # Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F2 = fftpack.fftshift( F1 )
        
        # Calculate a 2D power spectrum
        psd2D = np.abs( F2 )**2
        
        # Calculate the azimuthally averaged 1D power spectrum
        # psd1D = radialProfile.azimuthalAverage(psd2D)
        psd1D = azimuthalAverage(psd2D)
        
        # py.semilogy(psd1D)
        Spatial_Frequency = (np.arange(len(psd1D))+1)#*2/scale*(2*np.pi)
        plt.clf()
        plt.loglog(Spatial_Frequency, psd1D)
        py.xlabel('Spatial Frequency')
        py.ylabel('Power Spectrum')
        plt.pause(0.1)
        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(Spatial_Frequency[:160]), np.log10(psd1D[:160]))
        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(Spatial_Frequency[1000:]), np.log10(psd1D[1000:]))
        # print(slope)



OS = "Ubuntu"
# OS = "Windows"

if OS == "Ubuntu":
    chemin = "/home/moli/Bureau/Document_Ubuntu"
elif OS == "Windows":
    chemin = "D:/Travail/These/Document_Ubuntu"


def Film_w():
    
    matplotlib.use("Agg")
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    
    Start_image = 1
    N_image = 224
    fig = plt.figure(figsize=(8,6.2))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    pas_t = 1
    _ = Start_image
    
    with writer.saving(fig, "Turbulence_256_2048_15fps_farge_colormap.mp4", 100):
        while _ < N_image:
    
            print(_)
            
            iso_turb = np.fromfile(chemin+'/Code_cmm_cuda2d_Moli/data/vortex_shear_1000_4/all_save_data/w_'+str(_)+'.data', dtype='<f8', count=-1)
            scale = int(np.sqrt(len(iso_turb)))
            image = np.reshape(iso_turb, [scale, scale])
            
            plt.clf()
            # plt.imshow(image, origin = 'lower',  cmap ='jet', )
            plt.pcolormesh(image, cmap=farge_colormaps.farge_colormap_multi(type='vorticity'))
            plt.colorbar()
            # fig.colorbar(im)
            plt.axis('off')
            plt.xlim([0, scale])
            plt.ylim([0, scale])
            
            _ += pas_t
            writer.grab_frame()
            
    plt.close('all')





























