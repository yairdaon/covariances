from dolfin import *
import helper
import parameters

#########################################################
# Mixed Robin ###########################################
#########################################################
def mixed( container, mode ):
    
    beta = parameters.MixedRobin( container )

    u = container.u
    v = container.v
    normal = container.normal
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )
    

    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + 0.5*(abs(inner(beta,normal)) + inner(beta,normal))*u*v*ds
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( container, b )

    sol_mix_rob = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_mix_rob.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_mix_rob,
                       ["Mixed Robin", "Greens Function"],
                       container )
    
    helper.save_plots( container.variances( "mixed_robin" ),
                       ["Mixed Robin", "Variance"],
                       container )

    
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
    
    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx + 1.42*kappa*u*v*ds
    A = assemble(a)
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( container, b )

    sol_naive_rob = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_naive_rob.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_naive_rob, 
                       ["Naive Robin", "Greens Function"],
                       container )
   
    helper.save_plots( container.variances( "naive_robin" ),
                       ["Naive Robin", "Variance"],
                       container )
