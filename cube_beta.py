#!/usr/bin/python
import numpy as np

from dolfin import *

import container
import helper
import betas2D
from helper import dic as dic

mesh_obj = helper.get_refined_mesh( "cube", 
                                    nor = 2, # nor: Number Of Refinements
                                    tol = 0.2,
                                    factor = 0.35,
                                    betas = True )

container = container.Container( "cube", 
                                 mesh_obj,
                                 dic["cube"].kappa ) # == kappa == Killing rate

beta_expr = betas2D.Mix3D( container, "mix2" )

xyz = lambda s: ( 0.0, 0.5, s )

pt_list = np.linspace(0.0,1.0,106)
file_name = "../PriorCov/data/cube/beta.txt"
helper.empty_file( file_name )    
for s in pt_list:
    beta = -beta_expr( xyz(s) )[0]
    helper.add_point( file_name, s, beta )
