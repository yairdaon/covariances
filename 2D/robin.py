#!/usr/bin/python
from dolfin import *
import numpy as np
import helper
import parameters
import matplotlib.pyplot as plt
import pdb
import scipy as sp

# Choose a mesh
if False:
    mesh_name = "lshape"
    mesh_name = "dolfin_coarse"
    #mesh_name = "dolfin_fine"
    #mesh_name = "pinch" 
    mesh_obj = Mesh( "meshes/" + mesh_name )
else:
    mesh_name = "square"
    mesh_obj = UnitSquareMesh( 19, 19 )

# Variables that everybody uses.
kappa = 11. # Killing rate
dim = 2 # dimentia
k = 100#000 # number of iterations for ptwise var estimate


#########################################################
# Improper Homogeneous Robin ############################
nu = 0
container = parameters.Container( mesh_obj, kappa, dim, nu )
normal = container.normal 
u      = container.u
v      = container.v
tmp    = Function( container.V )
fund   = Function( container.V )
f      = Constant( 0.0 )

imp_beta   = parameters.Robin( container, param = "hom_beta" )

a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( imp_beta, normal )*u*v*ds
A = assemble(a)
L = f*v*dx
b = assemble(L)
helper.apply_sources( mesh_name, container, b )

sol_imp_rob = Function( container.V )
solve( A, tmp.vector(), b )
solve( A, sol_imp_rob.vector(), assemble(tmp*v*dx) )
imp_rob_var , _ = helper.get_var_and_g( A, container, k )
helper.save_plots( imp_rob_var, "Improper Robin Variance"       , mesh_name )
helper.save_plots( sol_imp_rob, "Improper Robin Greens Function", mesh_name )

##########################################################
# Homogeneous Robin ######################################
nu = 1
container  = parameters.Container( mesh_obj, kappa, dim, nu )

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
helper.apply_sources( mesh_name, container, b )

sol_hom_rob = Function( container.V )
solve( A, tmp.vector(), b )
solve( A, sol_hom_rob.vector(), assemble(tmp*v*dx) )
hom_rob_var , _ = helper.get_var_and_g( A, container, k )
helper.save_plots( hom_rob_var, "Homogeneous Robin Variance"       , mesh_name )
helper.save_plots( sol_hom_rob, "Homogeneous Robin Greens Function", mesh_name )


##########################################################
# Homogeneous Neumann ####################################
a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
A = assemble(a)
L = f*v*dx
b = assemble(L)
helper.apply_sources( mesh_name, container, b )

sol_neumann = Function( container.V )
solve( A, tmp.vector(), b )
solve( A, sol_neumann.vector(), assemble(tmp*v*dx) )
neumann_var , g = helper.get_var_and_g( A, container, k )

helper.save_plots( neumann_var, "Homogeneous Neumann Variance"       , mesh_name )
helper.save_plots( sol_hom_rob, "Homogeneous Neumann Greens Function", mesh_name )


############################################################
# Constant Variance / Time Change Method ###################
helper.apply_sources( mesh_name, container, b, scaling = g )

sol_cos_var = Function( container.V )
solve( A, tmp.vector(), b )
solve( A, sol_cos_var.vector(), assemble(tmp*v*dx) )

sol_cos_var.vector().set_local(  
    sol_cos_var.vector().array() * g.vector().array() 
) 

helper.save_plots( sol_cos_var, "Constant Variance Greens Function"  , mesh_name )


#############################################################
# Fundamental solution ######################################
fund_xpr = hom_beta.generate( "mat12" )
fund_xpr.x[0] = helper.pts[mesh_name][0][0]
fund_xpr.x[1] = helper.pts[mesh_name][0][1]
fund.interpolate( fund_xpr )
fund.vector()[:] = -fund.vector()[:]
helper.save_plots( fund, "Fundamental Solution", mesh_name )

interactive()

