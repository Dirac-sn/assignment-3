

import numpy as np
from Myintegration import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import legendre,eval_legendre
import sympy as sym


def func_int(f1,key,P,n=None):
    if key == 1:
        return lambda x: f1(x)*np.cos((2*n*np.pi*x)/P)
    elif key == 2:
        return lambda x: f1(x)*np.sin((2*n*np.pi*x)/P)

'''
def fourier_a(f,P,n,method,d,typef):
    a_n = np.zeros(n+1)
    b_n = np.zeros(n+1) 
    for i in range(0,n+1):
        if method == "t" and typef == 0 :
           a_n[i] = (2/P)*MyTrap(func_int(f,1,P,i), (-P/2),(P/2),1000,d)
        elif method == "s" and typef == 0 :
           a_n[i] = (2/P)*MySimp(func_int(f,1,P,i), (-P/2),(P/2),1000,d)       
        elif method == "q" and typef == 0 :
           a_n[i] = (2/P)*MyLegQuadrature(func_int(f,1,P,i), (-P/2),(P/2),1000,d)             
        elif method == "t" and typef == 1 :
           b_n[i] = (2/P)*MyTrap(func_int(f,2,P,i), (-P/2),(P/2),1000,d)
        elif method == "s" and typef == 1 :
           b_n[i] = (2/P)*MySimp(func_int(f,2,P,i), (-P/2),(P/2),1000,d)       
        elif method == "q" and typef == 1 :
           b_n[i] = (2/P)*MyLegQuadrature(func_int(f,2,P,i), (-P/2),(P/2),1000,d)         
        elif method == "t" and typef == -1 :
           a_n[i] = (2/P)*MyTrap(func_int(f,1,P,i), (-P/2),(P/2),1000,d)
           b_n[i] = (2/P)*MyTrap(func_int(f,2,P,i), (-P/2),(P/2),1000,d)
        elif method == "s" and typef == -1 :
           a_n[i] = (2/P)*MySimp(func_int(f,1,P,i), (-P/2),(P/2),1000,d)      
           b_n[i] = (2/P)*MySimp(func_int(f,2,P,i), (-P/2),(P/2),1000,d)       
        elif method == "q" and typef == -1 :
           a_n[i] = (2/P)*MyLegQuadrature(func_int(f,1,P,i), (-P/2),(P/2),1000,d)
           b_n[i] = (2/P)*MyLegQuadrature(func_int(f,2,P,i), (-P/2),(P/2),1000,d)       
    return [a_n,b_n]   '''
          

def fourier(f,P,n,d):
    
    a_n = np.zeros(n)
    b_n = np.zeros(n) 
    a0  = (1/P)*MyLegQuadrature(f,(-P/2),(P/2),1000,d)
    for i in range(1,n+1):
       a_n[i] = (2/P)*MyLegQuadrature(func_int(f,1,P,i),(-P/2),(P/2),1000,d)
       b_n[i] = (2/P)*MyLegQuadrature(func_int(f,2,P,i),(-P/2),(P/2),1000,d)  
   
    return a0,a_n,b_n

def func1(x):
    return x

coeff = fourier(func1, 2, 10,6)

print(coeff)

def partial_series(f,P,x,n,d):
    
    coeff = fourier(f,P,n,d)
    coss = np.zeros(n)
    sins = np.zeros(n)
    for i in range(n):
        coss[i] = np.cos((2*n*np.pi*x)/P)
        sins[i] = np.sin((2*n*np.pi*x)/P)
    
    sum1 = coeff[0] + coss.dot(coeff[1]) + sins.dot(coeff[2])
    return sum1

partial_series = np.vectorize(partial_series)


print(partial_series(func1,2,0.5,10,6))















        