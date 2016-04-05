from dolfin import *
import numpy as np
import helper

#########################################################
# Dirichlet #############################################
#########################################################
def dirichlet( container, mode, get_var ):

    def boundary(x, on_boundary):
        return on_boundary
        
    bc = DirichletBC(container.V,  Constant(0.0), boundary)


    u = container.u
    v = container.v
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )

    
    a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
    L = f*v*dx
        
    A, b = assemble_system ( a, L, bc )
    
    helper.apply_sources( container, b )

    sol_dirichlet = Function( container.V )
    solve( A, tmp.vector(), b )
    solve( A, sol_dirichlet.vector(), assemble(tmp*v*dx) )
    helper.save_plots( sol_dirichlet,
                       "Dirichlet Greens Function",
                       container.mesh_name,
                       ran = container.ran_sol,
                       mode = mode )    
    
    if get_var: 
        dirichlet_var = container.dirichlet_var
        helper.save_plots( dirichlet_var,
                           "Dirichlet Variance",
                           container.mesh_name,
                           ran = container.ran_var,
                           mode = mode )
    
