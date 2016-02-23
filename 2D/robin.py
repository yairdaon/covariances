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



# Improper Homogeneous Robin ######################################
nu = 0
container = parameters.Container( mesh_obj, kappa, dim, nu )
normal = container.normal 
u      = container.u
v      = container.v
tmp    = Function( container.V )
f      = Constant( 0.0 )

imp_beta   = parameters.Robin( container, param = "hom_beta" )

a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( imp_beta, normal )*u*v*ds
A = assemble(a)
L = f*v*dx
b = assemble(L)
helper.apply_sources( file_name, container, b )

sol_imp_rob = Function( container.V )
solve( A, tmp.vector(), b )
solve( A, sol_imp_rob.vector(), assemble(tmp*v*dx) )
imp_rob_var = helper.get_var( A, container, k )


# Homogeneous Robin ######################################
nu = 1
container  = parameters.Container( mesh_obj, kappa, dim, nu )
#ran = [ 0.0, 0.004 ]

normal = container.normal 
u      = container.u
v      = container.v
tmp    = Function( container.V )
f      = Constant( 0.0 )

hom_beta   = parameters.Robin( container, param = "hom_beta" )

a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( hom_beta, normal )*u*v*ds
A = assemble(a)
L = f*v*dx
b = assemble(L)
helper.apply_sources( file_name, container, b )

sol_hom_rob = Function( container.V )
solve( A, tmp.vector(), b )
solve( A, sol_hom_rob.vector(), assemble(tmp*v*dx) )
hom_rob_var = helper.get_var( A, container, k )


# Homogeneous Neumann ######################################
a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
A = assemble(a)
L = f*v*dx
b = assemble(L)
helper.apply_sources( file_name, container, b )

sol_neumann = Function( container.V )
solve( A, tmp.vector(), b )
solve( A, sol_neumann.vector(), assemble(tmp*v*dx) )
neumann_var = helper.get_var( A, container, k )


# Fundamental solution
fundamental = hom_beta.generate( "mat12" )
fundamental.x[0] = 0.45
fundamental.x[1] = 0.65
tmp.interpolate( fundamental )
tmp.vector()[:] = -tmp.vector()[:]


plot( hom_rob_var, 
      title = "Homogeneous Robin Variance",    
      mode ="color",
      #range_min = ran[0],
      #range_max = ran[1],
  ).write_png( "../../PriorCov/" + mesh_name + "_homogeneous_robin_variance" )

plot( neumann_var,
      title = "Neumann Variance", 
      mode ="color",
      #range_min = ran[0],
      #range_max = ran[1]
  ).write_png( "../../PriorCov/" + mesh_name + "_homogeneous_neumann_variance" )

plot( imp_rob_var,
      title = "Improper Robin Variance",
      mode ="color",
      #range_min = ran[0],
      #range_max = ran[1]
  ).write_png( "../../PriorCov/" + mesh_name + "_improper_robin_variance" )

plot( tmp,
      title = "Fundamental Solution",
      mode ="color",
      #range_min = ran[0],
      #range_max = container.factor
  ).write_png( "../../PriorCov/" + mesh_name + "_fundamental_solution" )

plot( sol_hom_rob, 
      title = "Homogeneous Robin Greens Function",
      mode = "color",
      #range_min = ran[0],
      #range_max = ran[1]
  ).write_png( "../../PriorCov/" + mesh_name + "_homogeneous_robin_green" )

plot( sol_imp_rob, 
      title = "Improper Greens Function",
      mode ="color" ,
      #range_min = ran[0],
      #range_max = ran[1]
  ).write_png( "../../PriorCov/" + mesh_name + "_improper_robin_green" )

plot( sol_neumann,
      title = "Neumann Greens Function",
      mode ="color",
      #range_min = ran[0],
      #range_max = ran[1]
  ).write_png( "../../PriorCov/" + mesh_name + "_homogeneous_neumann_green" )

#interactive()

