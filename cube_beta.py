#!/usr/bin/python
import scipy.integrate as spi
import scipy as sp
import numpy as np
import math
import sys
import time
import pdb
import os

from dolfin import *

import container
import helper
import mixed3D
   
kappa = 11.0
                  
mesh_name = "cube"
mesh_obj = helper.refine_cube( 67, 219, 87,nor = 0 )

                               # tol = .3,
                               # factor = .8,
                               # show = True,
                               # betas = True,
                               # slope = 0.4 )

container = container.Container( mesh_name,
                                 mesh_obj,
                                 kappa, # == kappa == Killing rate
                                 num_samples = 0 )

mixed3D.Mixed.value_shape = lambda x: ()
beta_expr = mixed3D.Mixed( container, normal_run = False )

x = lambda s: 0.0
y = lambda s: s
z = lambda s: 0.5

pt_list = np.linspace(0.05,0.95,87)
file_name = "data/cube/beta.txt"
try:
    os.remove( file_name )
except:
    pass
    
for s in pt_list:
    beta = beta_expr( x(s), y(s), z(s) )
    helper.add_point( file_name, s, beta )




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

