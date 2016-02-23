#!/usr/bin/python
from dolfin import *
import numpy as np
import helper
import parameters
import matplotlib.pyplot as plt
import pdb
import scipy as sp

# Choose a mesh
if True:
    mesh_name = "lshape"
    mesh_name = "dolfin_coarse"
    mesh_name = "dolfin_fine"
    #mesh_name = "pinch" 
    file_name = mesh_name + ".xml"
    mesh_obj = Mesh( "meshes/" + file_name )
else:
    mesh_name = "square"
    mesh_obj = UnitSquareMesh( 9, 9)

# Global variables. Everybody uses them.
kappa = 11. # Killing rate
dim = 2 # dimentia
k = 2000 # number of iterations for ptwise var estimate



# Preparations
nu = 1
container = parameters.Container( mesh_obj, kappa, dim, nu )
u      = container.u
v      = container.v
tmp    = Function( container.V )
f      = Constant( 0.0 )

# Homogeneous Neumann ######################################
a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
A = assemble(a)
L = f*v*dx
b = assemble(L)

g = helper.get_g( A, container, k )
helper.apply_sources( file_name, container, b, g = g )

sol = Function( container.V )
solve( A, tmp.vector(), b )
solve( A, sol.vector(), assemble(tmp*v*dx) )

sol.vector().set_local(  
    sol.vector().array() * g.vector().array() 
) 


# Fundamental solution
fundamental = container.generate( "mat12" )
fundamental.x[0] = 0.45
fundamental.x[1] = 0.65
tmp.interpolate( fundamental )
tmp.vector()[:] = -tmp.vector()[:]

plot( tmp,
      title = "Fundamental Solution",
      #mode ="color",
      #range_min = ran[0],
      #range_max = container.factor
  ).write_png( "../../PriorCov/" + mesh_name + "_fundamental_solution" )

plot( sol,
      title = "Neumann Greens Function",
      #mode ="color",
      #range_min = ran[0],
      #range_max = ran[1]
  ).write_png( "../../PriorCov/" + mesh_name + "_modified_neumann_green" )

interactive()

