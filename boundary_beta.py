#!/usr/bin/python
from dolfin import *
import sys
import time
import pdb

import container
import helper
import mixed3D
                     
mesh_name = "cube"
mesh_obj = helper.refine_cube( 20, 20, 20, 
                               nor = 0, 
                               tol = 0.0,
                               factor = 1.0,
                               #show = True,
                               greens = True )

container = container.Container( mesh_name,
                                 mesh_obj,
                                 5., # == kappa == Killing rate
                                 num_samples = 0 )

mixed3D.Mixed.value_shape = lambda x: ()
beta_expr = mixed3D.Mixed( container, normal_run = False )
beta_func = interpolate( beta_expr, container.V )

helper.save_plots( beta_func,
                   ["boundary beta"],
                   container )

# Maybe this will prevent further runs from crashing??
mixed3D.Mixed.value_shape = lambda x: (3,)


