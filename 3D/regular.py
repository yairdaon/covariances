from dolfin import *
import helper

def ordinary( container, mode, BC ):

    u = container.u
    v = container.v
    normal = container.normal
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )

    loc_solver = container.solvers( BC )
    L = f*v*dx
    b = assemble(L)
    helper.apply_sources( container, b )

    sol_mix_rob = Function( container.V )
    loc_solver( tmp.vector(), b )
    loc_solver( sol_mix_rob.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_mix_rob,
                       [ BC, "Greens Function"],
                       container )
    
    helper.save_plots( container.variances( BC ),
                       [ BC, "Variance"],
                       container )
