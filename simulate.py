#!/usr/bin/python
from dolfin import *
import time
import sys
import numpy as np
import scipy.special as sp

import helper
import container
from helper import dic as dic

def fundamental2D( cot ): 
    '''
    plots fundamental solution in 2D
    '''
    x = dic[cot.mesh_name].source
    
    V = cot.V
    mesh_obj = cot.mesh_obj
    kappa = cot.kappa
    
    y  = cot.mesh_obj.coordinates()

    x_y = x-y
    ra  = x_y * x_y
    ra  = np.sum( ra, axis = 1 )
    ra  = np.sqrt( ra ) + 1e-13
    kappara = kappa * ra

    phi_arr = cot.factor * np.power( kappara, 1.0 ) * sp.kn( 1, kappara )
    phi     = Function( V )
    phi.vector().set_local( phi_arr[dof_to_vertex_map(V)] )

    helper.save_plots( phi, 
                       ["Free Space", "Greens Function"], 
                       cot )

def fundamental3D( cot ):
    '''
    plots fundamental solution in 3D
    '''
    
    x = dic[cot.mesh_name].source
    
    V = cot.V
    mesh_obj = cot.mesh_obj
    kappa = cot.kappa
    
    y  = cot.mesh_obj.coordinates()

    x_y = x-y
    ra  = x_y * x_y
    ra  = np.sum( ra, axis = 1 )
    ra  = np.sqrt( ra ) + 1e-13
    kappara = kappa * ra

    phi_arr = cot.factor * np.power( kappara, 0.5 ) * sp.kv( 0.5, kappara )
    phi     = Function( V )
    phi.vector().set_local( phi_arr[dof_to_vertex_map(V)] )

    helper.save_plots( phi, 
                       ["Free Space", "Greens Function"], 
                       cot )


def fundamental( cot ):
    ''' 
    pretty self explanatory.
    '''
    if cot.dim == 2:
        fundamental2D( cot )
    if cot.dim == 3:
        fundamental3D( cot )


def green( cot, BC ):
    '''
    plots the domain Green's function with boundary
    condition BC on the mesh for a source.
    '''
    u = cot.u
    v = cot.v
    normal = cot.normal
    kappa = cot.kappa
    f = Constant( 0.0 )
    tmp = Function( cot.V )

    loc_solver = cot.solvers( BC )
    L = f*v*dx
    b = assemble(L)
    
    # Make the RHS have delta funcitons at the sources
    helper.apply_sources( cot, b )

    sol = Function( cot.V )
    loc_solver( tmp.vector(), b )
    loc_solver( sol.vector(), assemble(tmp*v*dx) )
    
    # Create the descrition list so we keep track of files
    desc = [ BC, "Greens Function" ]
    if "ours" in BC:
        desc.append( cot.quad )
        
    helper.save_plots( sol,
                       desc,
                       cot )
    

def variance( cot, BC ):
    '''
    plots domain pointwise variance with boundary condition BC.
    '''
    u = cot.u
    v = cot.v
    kappa = cot.kappa
    f = Constant( 0.0 )
    tmp = Function( cot.V )
    
    loc_solver = cot.solvers( BC )

    L = f*v*dx
    b = assemble(L)
    if "dirichlet" in BC:
        pass
    else:
        g = cot.gs( BC )
        helper.apply_sources( cot, b, scaling = g )

        sol_constant_var = Function( cot.V )
        loc_solver( tmp.vector(), b )
        loc_solver( sol_constant_var.vector(), assemble(tmp*v*dx) )
        sol_constant_var.vector().set_local(  
            sol_constant_var.vector().array() * g.vector().array() 
        ) 
    
    # Create the descrition list so we keep track of files
    greens_desc = [ BC, "Constant Variance", "Greens Function"]
    std_desc    = [ BC, "Standard Deviation" ]
    if "ours" in BC:
        greens_desc.append( cot.quad )
        std_desc.append( cot.quad )

    helper.save_plots( cot.stds( BC ),
                       std_desc,
                       cot )

    if not "dirichlet" in BC:
        helper.save_plots( sol_constant_var,
                           greens_desc,
                           cot )
        
  

if __name__ == "__main__":

    mesh_name = sys.argv[1]
    if len( sys.argv ) > 2 :
        quad = sys.argv[2]
    else:
        quad = "std"
    print
    print mesh_name

    if "cube" in mesh_name:
        numSamples = 10000
    else:
        numSamples = 0
    
    cot = container.Container( mesh_name,
                               dic[mesh_name](), # get the mesh
                               dic[mesh_name].alpha,
                               gamma = 1,
                               quad = quad,
                               numSamples = numSamples )
    print "fundamental"
    start_time = time.time()
    fundamental( cot )
    print "Run time: " + str( time.time() - start_time )
    print

    print "dirichlet"
    start_time = time.time()
    green( cot, "dirichlet" )
    print "Run time: " + str( time.time() - start_time )
    print

    print "neumann"
    start_time = time.time()
    green( cot, "neumann" )
    print "Run time: " + str( time.time() - start_time )
    print
    
    if not "square" in mesh_name: 
        
        print "ours"
        start_time = time.time()
        green( cot, "ours" )
        print "Run time: " + str( time.time() - start_time )
        
        print "our robin variance"
        start_time = time.time()
        variance( cot, "ours" )
        print "Run time: " + str( time.time() - start_time )
        print

        print "neumann variance"
        start_time = time.time()
        variance( cot, "neumann" )
        print "Run time: " + str( time.time() - start_time )
        print
        
        print "roininen" 
        start_time = time.time()
        green( cot, "roininen" )
        print "Run time: " + str( time.time() - start_time )
        print
        
        print "dirichlet variance"
        start_time = time.time()
        variance( cot, "dirichlet" )
        print "Run time: " + str( time.time() - start_time )
        print

        print "roininen variance" 
        start_time = time.time()
        variance( cot, "roininen" )
        print "Run time: " + str( time.time() - start_time )
        print       


    
