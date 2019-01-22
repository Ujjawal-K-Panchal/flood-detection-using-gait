# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:46:33 2018

@author: Uchiha Madara
Note! : Please read comments!!!
"""

import numpy as np
from scipy.fftpack import rfft

#Below function is for l1 of length 50.
#I donot understand the relation between length of l1 and n. (Order of fft)

def spectral_energy(l1):
    n = 16 #Assuming Window size of 50
    Y = rfft(l1,n)
    P = abs(Y/n)
    P = P[1 : ( (n//2)+1 )] # double // to get integer equivalent.
    energy = np.sum(np.power(P,2))
    return energy
    
    
    
    