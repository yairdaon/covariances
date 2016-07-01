from scipy import special as sp
import numpy as np

from dolfin import *

import helper
from helper import dic as dic

#########################################################
# Fundamental solution ##################################
#########################################################
def fundamental( container ):
    
    x = dic[container.mesh_name].source
    
    V = container.V
    mesh_obj = container.mesh_obj
    kappa = container.kappa
    
    y  = container.mesh_obj.coordinates()

    x_y = x-y
    ra  = x_y * x_y
    ra  = np.sum( ra, axis = 1 )
    ra  = np.sqrt( ra ) + 1e-13
    kappara = kappa * ra

    phi_arr = container.factor * np.power( kappara, 0.5 ) * sp.kv( 0.5, kappara )
    phi     = Function( V )
    phi.vector().set_local( phi_arr[dof_to_vertex_map(V)] )

    helper.save_plots( phi, 
                       ["Free Space", "Greens Function"], 
                       container )
