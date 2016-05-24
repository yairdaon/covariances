from dolfin import *
import numpy as np

import helper



def variance( container, mode, BC ):

    u = container.u
    v = container.v
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )
    g = container.gs( BC )
    
    loc_solver = container.solvers( BC )

    L = f*v*dx
    b = assemble(L)

    helper.apply_sources( container, b, scaling = g )

    sol_cos_var = Function( container.V )
    loc_solver( tmp.vector(), b )
    loc_solver( sol_cos_var.vector(), assemble(tmp*v*dx) )
    
    sol_cos_var.vector().set_local(  
        sol_cos_var.vector().array() * g.vector().array() 
    ) 
    
    helper.save_plots( sol_cos_var,
                       [ BC + " Constant Variance", "Greens Function"],
                       container )
    
    helper.save_plots( container.variances( BC ),
                       [ BC, "Variance"],
                       container )
