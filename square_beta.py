#!/usr/bin/python
import scipy.integrate as spi
import scipy as sp
import numpy as np
import math
import sys
import time
import pdb
import os
import matplotlib.pyplot as plt

from dolfin import *

import container
import helper
import betas2D
   
kappa = 5.0
                  
mesh_name = "square"
mesh_obj = UnitSquareMesh( 673, 673 )

container = container.Container( mesh_name,
                                 mesh_obj,
                                 kappa, # == kappa == Killing rate
                                 num_samples = 0 )

imp_beta = betas2D.Beta( container, "imp_enum", "imp_denom" )
mix_beta = betas2D.Beta( container, "mix_enum", "mix_denom" )

x = lambda s: 0.0
y = lambda s: s

pt_list = np.linspace(0.05,0.95,87)
imp_beta_file = "../PriorCov/imp_beta.txt"
mix_beta_file = "../PriorCov/mix_beta.txt"
try:
    os.remove( imp_beta_file )
except:
    pass
try:
    os.remove( mix_beta_file )
except:
    pass

imp_list = []
mix_list = []
for s in pt_list:
    imp = -imp_beta( x(s), y(s) )[0]
    helper.add_point( imp_beta_file, s, imp )
    imp_list.append( imp )
    mix = -mix_beta( x(s), y(s) )[0]
    helper.add_point( mix_beta_file, s, mix )
    mix_list.append( mix )

plt.plot( pt_list, imp_list, color = "r" )
plt.plot( pt_list, mix_list, color = "b" )
plt.show()


# def enum(z,y,x, kappa, h):
    
#     ra = np.sqrt( x*x + y*y + z*z ) + 1e-13
#     kappara = kappa * ra 
#     tmp = kappa**2 * (2 + np.power( kappara, -1 )) * np.power( kappara, -1.5 ) * np.exp( -kappara ) * sp.special.kv( 0.5, kappara ) * x 
    
#     return np.sum(tmp) * h**3

# def denom(z,y,x, kappa, h):
#     ra = np.sqrt( x*x + y*y + z*z ) + 1e-13
#     kappara = kappa * ra 
#     tmp = 2. * sp.special.kv( 0.5, kappara ) * np.exp( -kappara ) * np.power( kappara, -0.5)
#     return np.sum(tmp) * h**3

# h = 0.024145
# x = np.arange(   0, 1.0, h )   
# y = np.arange( -.5,  .5, h )
# z = np.arange( -.5,  .5, h )
# X, Y, Z = np.meshgrid( x, y ,z )
# print enum(X,Y,Z,kappa,h) / denom(Z, Y, X, kappa, h) 

