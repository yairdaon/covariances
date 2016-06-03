#!/usr/bin/python
import scipy.integrate as spi
import scipy as sp
import numpy as np
import math
import sys
import time
import pdb

from dolfin import *

import container
import helper
import mixed3D
                     
mesh_name = "cube"
mesh_obj = helper.refine_cube( 5, 5, 5, 
                               nor = 4, 
                               tol = .25,
                               factor = .8,
                               #show = True,
                               greens = True )

container = container.Container( mesh_name,
                                 mesh_obj,
                                 5., # == kappa == Killing rate
                                 num_samples = 0 )

mixed3D.Mixed.value_shape = lambda x: ()
beta_expr = mixed3D.Mixed( container, normal_run = False )
print beta_expr( (0, 0.5, 0.5) )





K_half = lambda x: sp.special.kv( 0.5, x )

def enum(z,y,x):
    kappa = 5.
    v = (x,y,z)
    r = kappa * np.linalg.norm( v ) + 1e-13
    
    return kappa**2 * (2 + 1./r) * math.exp( -r ) * r**(-1.5) * K_half( r ) * x

def denom(z,y,x):
    kappa = 5.
    v = (x,y,z)
    r = kappa * np.linalg.norm( v ) + 1e-13 
    
    return 2. * r**(-0.5) * K_half( r ) * math.exp( -r )

top = 0.5
bot = -top
a =  0.
b =  1.0
gfun = lambda x: bot
hfun = lambda x: top
qfun = lambda x,y: bot
rfun = lambda x,y: top
    
enum_num  = spi.tplquad( enum , a, b, gfun, hfun, qfun, rfun, epsabs=1.49e-03, epsrel=1.49e-03 )
denom_num = spi.tplquad( denom, a, b, gfun, hfun, qfun, rfun, epsabs=1.49e-03, epsrel=1.49e-03 )

print enum_num[0] /denom_num[0]



# beta_func = interpolate( beta_expr, container.V )

# helper.save_plots( beta_func,
#                    ["boundary beta"],
#                    container )

# # Maybe this will prevent further runs from crashing??
# mixed3D.Mixed.value_shape = lambda x: (3,)


