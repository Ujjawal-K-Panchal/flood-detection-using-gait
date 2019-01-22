# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 19:50:58 2018

@author: uchih
"""
import numpy as np
from scipy.fftpack import rfft

def sum_fft(l1):
    coeffs = rfft(l1)
    sum_fft = np.sqrt( np.sum( np.power( coeffs[0:5] , 2 ) ) )
    return sum_fft

l1 = list([1,2,3,4,5,6,7,8,9,10])

print(sum_fft(l1))
