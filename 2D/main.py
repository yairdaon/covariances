#!/usr/bin/python
from dolfin import *

import helper
import parameters
import neumann
import robin
import fundamental
import variance

# Choose a mesh
if True:
    mesh_name = "lshape"
    mesh_name = "dolfin_coarse"
    #mesh_name = "dolfin_fine"
    #mesh_name = "pinch" 
    mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )
else:
    mesh_name = "square"
    mesh_obj = UnitSquareMesh( 30, 30 )

kappa = 11. # Killing rate
dim = 2 # dimension
nu = 1

container = parameters.Container( mesh_name, mesh_obj, kappa, dim, nu )

#plotting mode.
#mode = color # this is flattend
mode = "auto" # this is 3D

neumann.neumann( container, mode )
variance.variance( container, mode )
robin.robin( container, mode )
robin.improper( container, mode )
fundamental.fundamental( container, mode )

interactive()

