from dolfin import *
import helper
import numpy as np

#########################################################
# Constant Variance / Time Change Method ################
#########################################################
def variance( container, mode ):

    u = container.u
    v = container.v
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )
    g = container.g

    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)

    helper.apply_sources( container, b, scaling = g )

    sol_cos_var = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_cos_var.vector(), assemble(tmp*v*dx) )
    
    sol_cos_var.vector().set_local(  
        sol_cos_var.vector().array() * g.vector().array() 
    ) 
    
    helper.save_plots( sol_cos_var,
                       "Constant Variance Greens Function",
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode )

