#!/usr/bin/python
import scipy.integrate as spi
import scipy as sp
import numpy as np
import math
import sys
import time


from dolfin import *

import container
import helper
import betas2D



print
print "Parallelogram beta"            

mesh_name = "parallelogram"
mesh_obj = helper.refine( mesh_name,
                          nor = 1,
                          tol = 0.35,
                          factor = 0.66 )

container = container.Container( mesh_name,
                                 mesh_obj,
                                 11. # == kappa == Killing rate
                             )
   
imp_beta = betas2D.Beta( container, "imp" )
mix_beta = betas2D.Beta( container, "mix" )

x = lambda s: s
y = lambda s: 2.5 * s

# Outward pointing normal
nn = np.array( [-2.5, 1.0] ) / math.sqrt( 1 + 2.5**2 )

pt_list = np.linspace(0.01,0.99,87)
imp_beta_file = "data/parallelogram/imp_beta.txt"
mix_beta_file = "data/parallelogram/mix_beta.txt"
helper.empty_file( imp_beta_file, mix_beta_file )

# imp_list = []
# mix_list = []
for s in pt_list:
   
    imp = imp_beta( x(s), y(s) )[0] * nn[0] +  imp_beta( x(s), y(s) )[1] * nn[1] 
    helper.add_point( imp_beta_file, s, imp )
    # imp_list.append( imp )
   
    mix = mix_beta( x(s), y(s) )[0] * nn[0] +  mix_beta( x(s), y(s) )[1] * nn[1] 
    helper.add_point( mix_beta_file, s, mix )
    # mix_list.append( mix )
