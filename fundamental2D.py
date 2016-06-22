from dolfin import *
import numpy as np
from scipy import special as sp

import helper

#########################################################
# Fundamental solution ##################################
#########################################################
def fundamental( container ):

    x = helper.pts[container.mesh_name]
    
    V = container.V
    mesh_obj = container.mesh_obj
    kappa = container.kappa
    
    y  = container.mesh_obj.coordinates()

    x_y = x-y
    ra  = x_y * x_y
    ra  = np.sum( ra, axis = 1 )
    ra  = np.sqrt( ra ) + 1e-13
    kappara = kappa * ra

    phi_arr = container.factor * np.power( kappara, 1.0 ) * sp.kv( 1.0, kappara )
    phi     = Function( V )
    phi.vector().set_local( phi_arr[dof_to_vertex_map(V)] )

    helper.save_plots( phi, 
                       ["Free Space", "Greens Function"], 
                       container )


