from dolfin import *
import helper
import parameters

#########################################################
# Homogeneous Robin #####################################
#########################################################
def robin( container ):
    
    hom_beta   = parameters.Robin( container, param = "hom_beta" )

    u = container.u
    v = container.v
    normal = container.normal
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )
    

    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( hom_beta, normal )*u*v*ds
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( container, b )

    sol_hom_rob = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_hom_rob.vector(), assemble(tmp*v*dx) )
    hom_rob_var , _ = helper.get_var_and_g( container, A )
    
    helper.save_plots( hom_rob_var, "Homogeneous Robin Variance"       , container.mesh_name, ran = container.ran_var )
    helper.save_plots( sol_hom_rob, "Homogeneous Robin Greens Function", container.mesh_name, ran = container.ran_sol )


#########################################################
# Improper Homogeneous Robin ############################
#########################################################
def improper( container ):
    cot = parameters.Container( container.mesh_name,
                                container.mesh_obj,
                                container.kappa,
                                container.dim, 
                                0 )# the only difference
 
    normal = cot.normal 
    u      = cot.u
    v      = cot.v
    tmp    = Function( cot.V )
    kappa  = cot.kappa
    f      = Constant( 0.0 )

    imp_beta   = parameters.Robin( cot, param = "hom_beta" )

    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + inner( imp_beta, normal )*u*v*ds
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( cot, b )

    sol_imp_rob = Function( cot.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_imp_rob.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_imp_rob, "Improper Robin Greens Function", cot.mesh_name, ran = container.ran_sol )
    
