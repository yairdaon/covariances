from dolfin import *
import helper

def ordinary( container, BC ):

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

    sol = Function( container.V )
    loc_solver( tmp.vector(), b )
    loc_solver( sol.vector(), assemble(tmp*v*dx) )
    
    helper.save_plots( sol,
                       [ BC, "Greens Function"],
                       container )
    
