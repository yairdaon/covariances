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


container = container.Container( mesh_name,
                                 mesh_obj,
                                 kappa, # == kappa == Killing rate
                                 num_samples = 0 )

beta_expr = mixed3D.Mixed( container )

x = lambda s: 0.0
y = lambda s: s
z = lambda s: 0.5

pt_list = np.linspace(0.05,0.95,113)
file_name = "data/cube/beta.txt"
helper.empty_file( file_name )    
for s in pt_list:
    beta = -beta_expr( x(s), y(s), z(s) )[0]
    helper.add_point( file_name, s, beta )
