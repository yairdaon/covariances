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
    kappa2 = container.kappa2
    gamma = container.gamma
    f = Constant( 0.0 )
    tmp = Function( container.V )
    g = container.gs( "neumann" )

    a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx
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
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )



#########################################################
# Naive Robin Constant Variance / Time Change Method ####
#########################################################
def naive_robin_variance( container, mode ):

    u = container.u
    v = container.v
    kappa = container.kappa
    kappa2 = container.kappa2
    gamma = container.gamma
    f = Constant( 0.0 )
    tmp = Function( container.V )
    g = container.gs( "naive_robin" )

    a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + 1.42*kappa*u*v*ds
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
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )



#########################################################
# Improper Robin Constant Variance / Time Change Method #
#########################################################
def improper_robin_variance( container, mode ):

    u = container.u
    v = container.v
    kappa2 = container.kappa2
    gamma = container.gamma
    normal = container.normal
    f = Constant( 0.0 )
    tmp = Function( container.V )
    g = container.gs( "improper_robin" )
    
    imp_beta = parameters.Robin( container, "imp_enum", "imp_denom" )
    a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + inner( imp_beta, normal )*u*v*ds
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
                       ["Improper Robin Constant Variance", "Greens Function"],
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )


#########################################################
# Mixed Robin Constant Variance / Time Change Method ####
#########################################################
def mixed_robin_variance( container, mode ):

    u = container.u
    v = container.v
    kappa2 = container.kappa2
    gamma = container.gamma
    normal = container.normal
    f = Constant( 0.0 )
    tmp = Function( container.V )
    g = container.gs( "mixed_robin" )
    
    mix_beta = parameters.Robin( container, "mix_enum", "mix_denom" )
    a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + inner( mix_beta, normal )*u*v*ds
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
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )

