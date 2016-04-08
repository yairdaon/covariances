from dolfin import *
import helper
import parameters

#########################################################
# Mixed Robin ###########################################
#########################################################
def mixed( container, mode ):
    
    beta = parameters.Robin( container, "mix_enum", "mix_denom" )

    u = container.u
    v = container.v
    normal = container.normal
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )
    

    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( beta, normal )*u*v*ds
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( container, b )

    sol_mix_rob = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_mix_rob.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_mix_rob,
                       "Mixed Robin Greens Function",
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )
    
    if "square" in container.mesh_name or "parallelogram" in container.mesh_name:
         pass
    else:
        mix_rob_var , _ = helper.get_var_and_g( container, A )
        helper.save_plots( mix_rob_var,
                           "Mixed Robin Variance",
                           container.mesh_name,
                           ran = container.ran_var,
                           mode = mode )
    

#########################################################
# Improper Homogeneous Robin ############################
#########################################################
def improper( container, mode ):
   
    container.power = 1
    container.set_constants()

    normal = container.normal 
    u      = container.u
    v      = container.v
    tmp    = Function( container.V )
    kappa  = container.kappa
    f      = Constant( 0.0 )

    imp_beta = parameters.Robin( container, "imp_enum", "imp_denom" )

    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( imp_beta, normal )*u*v*ds
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( container, b )

    sol_imp_rob = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_imp_rob.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_imp_rob, 
                       "Improper Robin Greens Function",
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )

    
    if "square" in container.mesh_name or "parallelogram" in container.mesh_name:
         pass
    else:
        imp_rob_var , _ = helper.get_var_and_g( container, A )
        helper.save_plots( imp_rob_var,
                           "Improper Robin Variance",
                           container.mesh_name,
                           ran = container.ran_var,
                           mode = mode )
    
    container.power = 2
    container.set_constants()

#########################################################
# Naive Robin ###########################################
#########################################################
def naive( container, mode ):
     
    normal = container.normal 
    u      = container.u
    v      = container.v
    tmp    = Function( container.V )
    kappa  = container.kappa
    f      = Constant( 0.0 )
    
    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + kappa*u*v*ds
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( container, b )

    sol_naive_rob = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_naive_rob.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_naive_rob, 
                       "Naive Robin Greens Function",
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )

    if "square" in container.mesh_name or "parallelogram" in container.mesh_name:
         pass
    else:
        naive_rob_var , _ = helper.get_var_and_g( container, A )
        helper.save_plots( naive_rob_var,
                           "Naive Robin Variance",
                           container.mesh_name,
                           ran = container.ran_var,
                           mode = mode )
    
