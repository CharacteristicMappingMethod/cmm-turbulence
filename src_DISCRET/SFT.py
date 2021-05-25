# -*- coding: utf-8 -*-
"""
Created on Sat May 15 17:17:12 2021

@author: Moli
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy


plt.close('all')

def grid():
    
    N = 10000
    
    x = np.linspace(0, 2*np.pi, N)
#    f = np.sin(x)+2*np.sin(4*x)+1/2*np.sin(6*x)+np.sin(30*x)
    f = 0*x
    for _ in range(1, 10):
        f += np.sin(x*_**2)/_
    
#    plt.figure(1)
    
#    plt.plot(f)
    
    f_fft = scipy.fft(f)
    f_ifft = scipy.ifft(scipy.fft(f))
    
#    plt.plot(np.real(f_ifft))
    
    plt.figure(2)
    
    plt.plot(abs(f_fft**2))
    
    
    
    my_dft = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            my_dft[k] += f[n]*(np.exp(-1.j*2*np.pi*k*n/N))
    
    
    plt.figure(2)
    
    plt.plot(abs(my_dft**2))
    
    
    
    my_idft = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            my_idft[k] += my_dft[n]*(np.exp(1.j*2*np.pi*k*n/N))
    my_idft = my_idft/N
    
#    plt.figure(1)
#    
#    plt.plot(np.real(my_idft))
    


def grid_rand():
    
    N = 1000
    
    x = np.random.random(N)*2*np.pi
    x = x - np.min(x)
#    x = np.arange(N)/N*2*np.pi
    x = np.sort(x)
    mid = np.zeros(N+1)
    mid[1:-1] = (x[1:]+x[:-1])/2
    mid[0] = (x[-1]-2*np.pi+x[0])/2
    mid[-1] = (x[-1]+2*np.pi+x[0])/2
    vol =  (mid[1:]-mid[:-1])
#    f = np.sin(x)+2*np.sin(4*x)+1/2*np.sin(6*x)+np.sin(30*x)
    f = 0*x
    for _ in range(1, 999):
        f += np.sin(x*_**2)/_
    
    my_dft = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
#            my_dft[k] += f[n]*(np.exp(-1.j*2*np.pi*x[k]*N/(2*np.pi)*x[n]/(2*np.pi)))
            my_dft[k] += f[n]*(np.exp(-1.j*2*np.pi*x[k]*N/(2*np.pi)*x[n]/(2*np.pi)))*vol[n]/(2*np.pi)
    
    plt.figure(2)
    
    plt.plot(x/(2*np.pi)*N, abs(my_dft**2))
    
    my_idft = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            my_idft[k] += my_dft[n]*(np.exp(1.j*2*np.pi*x[k]*N/(2*np.pi)*x[n]/(2*np.pi)))
#    my_idft = my_idft/N
    
    plt.figure(1)
    
    plt.plot(x, np.real(my_idft))
    plt.plot(x, f, '--', c='k')
    





















