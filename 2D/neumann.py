from dolfin import *
import numpy as np
import helper

#########################################################
# Homogeneous Neumann ###################################
#########################################################
def neumann( container ):

    u = container.u
    v = container.v
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )

    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( container, b )
        
    sol_neumann = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_neumann.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_neumann, "Homogeneous Neumann Greens Function", container.mesh_name, ran = container.ran_sol )    

    neumann_var = container.neumann_var
    helper.save_plots( neumann_var, "Homogeneous Neumann Variance"       , container.mesh_name, ran = container.ran_var )
    
