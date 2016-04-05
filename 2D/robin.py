from dolfin import *
import helper
import parameters

#########################################################
# Mixed Robin ###########################################
#########################################################
def mixed( container, mode, get_var ):
    
    beta = parameters.Robin( container, "hom_enum", "hom_denom" )

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

    sol_hom_rob = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_hom_rob.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_hom_rob,
                       "Mixed Robin Greens Function",
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )

    if get_var:
        hom_rob_var , _ = helper.get_var_and_g( container, A )
        helper.save_plots( hom_rob_var,
                           "Mixed Robin Variance",
                           container.mesh_name,
                           ran = container.ran_var,
                           mode = mode )
    

#########################################################
# Improper Homogeneous Robin ############################
#########################################################
def improper( container, mode, get_var ):
   
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

    if get_var:
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
def naive( container, mode, get_var):
     
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

    if get_var:
        naive_rob_var , _ = helper.get_var_and_g( container, A )
        helper.save_plots( naive_rob_var,
                           "Naive Robin Variance",
                           container.mesh_name,
                           ran = container.ran_var,
                           mode = mode )
    


#########################################################
# Integrated Robin ######################################
#########################################################
def integrated_robin( container, mode, get_var ):
    
    container.power = 1
    container.set_constants()

    normal = container.normal 
    u      = container.u
    v      = container.v
    tmp    = Function( container.V )
    kappa  = container.kappa
    f      = Constant( 0.0 )

    int_beta   = parameters.Robin( container, "int_enum", "int_denom" )

    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( int_beta, normal )*u*v*ds
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( container, b )

    sol_int_rob = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_int_rob.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_int_rob, 
                       "Integrated Robin Greens Function",
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )
    if get_var:
        int_rob_var , _ = helper.get_var_and_g( container, A )
        helper.save_plots( int_rob_var,
                           "Integrated Robin Variance",
                           container.mesh_name,
                           ran = container.ran_var,
                           mode = mode )

    container.power = 2
    container.set_constants()
     
