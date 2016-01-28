#!/usr/bin/python
from dolfin import *
import numpy as np
import helper
import parameters
import matplotlib.pyplot as plt
import pdb

# Choose a mesh
if True:
    file_name = "lshape.xml"
    file_name = "dolfin_coarse.xml"
    #file_name = "dolfin_fine.xml"
    #file_name = "pinch.xml" 
    mesh_obj = Mesh( "meshes/" + file_name )
else:
    file_name = "square"
    mesh_obj = UnitSquareMesh( 24, 24 )

kappa = 3.23
dim = 2
nu = 1
container  = parameters.Container( mesh_obj, kappa, dim, nu )


# All the pbjects used in the variational formulation below
normal = container.normal 
u      = container.u
v      = container.v
tmp    = Function( container.V )
f      = Constant( 0.0 )



# Homogeneous Robin ######################################
hom_beta   = parameters.Robin( container, param = "hom_beta" )

a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( hom_beta, normal )*u*v*ds
A = assemble(a)
L = f*v*dx
b = assemble(L)
helper.apply_sources( file_name, container, b )

sol_hom_rob = Function( container.V )
solve( A, tmp.vector(), b, 'lu')
solve( A, sol_hom_rob.vector(), tmp.vector(), 'lu')



# Inhomogeneous Robin ####################################
inhom_beta = parameters.Robin( container, param = "inhom_beta" )
g          = parameters.Robin( container, param = "g"          )

a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( inhom_beta, normal )*u*v*ds
A = assemble(a)
L = f*v*dx + inner( g, normal )*v*ds
b = assemble(L)
helper.apply_sources( file_name, container, b )

sol_inhom_rob = Function( container.V )
solve( A, tmp.vector(), b, 'lu')
solve( A, sol_inhom_rob.vector(), tmp.vector(), 'lu')





# All the pbjects used in the variational formulation below
nu = 0
container  = parameters.Container( mesh_obj, kappa, dim, nu )
normal = container.normal 
u      = container.u
v      = container.v
tmp    = Function( container.V )
f      = Constant( 0.0 )



# Improper Homogeneous Robin ######################################
imp_beta   = parameters.Robin( container, param = "hom_beta" )

a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( imp_beta, normal )*u*v*ds
A = assemble(a)
L = f*v*dx
b = assemble(L)
helper.apply_sources( file_name, container, b )

sol_imp_rob = Function( container.V )
solve( A, tmp.vector(), b, 'lu')
solve( A, sol_imp_rob.vector(), tmp.vector(), 'lu')

# Neumann ######################################
a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
A = assemble(a)
L = f*v*dx
b = assemble(L)
helper.apply_sources( file_name, container, b )

sol_neumann = Function( container.V )
solve( A, tmp.vector(), b, 'lu')
solve( A, sol_neumann.vector(), tmp.vector(), 'lu')




fundamental = hom_beta.generate( "mat12" )
fundamental.x[0] = 0.45
fundamental.x[1] = 0.65
tmp.interpolate( fundamental )
tmp.vector()[:] = -tmp.vector()[:]


# difference
fi = File( 'vis/fundamental.pvd' )
fi << tmp 
fi = File( 'vis/homogeneous.pvd' )
fi << sol_hom_rob
fi = File( 'vis/inhomogeneous.pvd' )
fi << sol_inhom_rob
fi = File( 'vis/improper.pvd' )
fi << sol_imp_rob
fi = File( 'vis/neumann.pvd' )
fi << sol_neumann


plot( tmp,           title = "fundamental solution" )
plot( sol_hom_rob,   title = "homogeneous robin"   ) 
plot( sol_inhom_rob, title = "inhomogeneous robin" ) 
plot( sol_imp_rob,   title = "improper covariance" )

# plot( sol_hom_rob - sol_inhom_rob,   title = "homogeneous - inhomogeneous"   ) 
# plot( sol_hom_rob - sol_imp_rob,     title = "homogeneous - improper"        )
# plot( sol_inhom_rob - sol_imp_rob,   title = "inhomogeneous - improper"      )

# The minus sign is already built into the expression!!!
plot( sol_hom_rob - tmp,   title = "homogeneous - fundamental"   ) 
plot( sol_imp_rob - tmp,   title = "improper - fundamental"        )
plot( sol_inhom_rob - tmp, title = "inhomogeneous - fundamental"      )
plot( sol_neumann - tmp, title = "inhomogeneous - fundamental"      )

interactive()
