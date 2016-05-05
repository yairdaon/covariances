from dolfin import *
import helper
import numpy as np
import parameters

#########################################################
# Neumann Constant Variance / Time Change Method ########
#########################################################
def neumann_variance( container, mode ):

    u = container.u
    v = container.v
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )
    g = container.gs( "neumann" )

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
                       ["Neumann Constant Variance", "Greens Function"],
                       container )



#########################################################
# Naive Robin Constant Variance / Time Change Method ####
#########################################################
def naive_robin_variance( container, mode ):

    u = container.u
    v = container.v
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )
    g = container.gs( "naive_robin" )

    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + 1.42*kappa*u*v*ds
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
                       ["Naive Robin Constant Variance", "Greens Function"],
                       container )

#########################################################
# Mixed Robin Constant Variance / Time Change Method ####
#########################################################
def mixed_robin_variance( container, mode ):

    u = container.u
    v = container.v
    kappa2 = container.kappa2
    normal = container.normal
    f = Constant( 0.0 )
    tmp = Function( container.V )
    g = container.gs( "mixed_robin" )
    
    beta = parameters.MixedRobin( container )
    a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx + 0.5*(abs(inner(beta,normal)) + inner(beta,normal))*u*v*ds
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
                       ["Mixed Robin Constant Variance", "Greens Function"],
                       container )

