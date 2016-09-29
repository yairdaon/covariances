#!/usr/bin/python
from dolfin import *
import time
import sys
import numpy as np
import scipy.special as sp

import helper
import container
from helper import dic as dic

def fundamental2D( container ): 
    '''
    plots fundamental solution in 2D
    '''
    x = dic[container.mesh_name].source
    
    V = container.V
    mesh_obj = container.mesh_obj
    kappa = container.kappa
    
    y  = container.mesh_obj.coordinates()

    x_y = x-y
    ra  = x_y * x_y
    ra  = np.sum( ra, axis = 1 )
    ra  = np.sqrt( ra ) + 1e-13
    kappara = kappa * ra

    phi_arr = container.factor * np.power( kappara, 1.0 ) * sp.kn( 1, kappara )
    phi     = Function( V )
    phi.vector().set_local( phi_arr[dof_to_vertex_map(V)] )

    helper.save_plots( phi, 
                       ["Free Space", "Greens Function"], 
                       container )

def fundamental3D( container ):
    '''
    plots fundamental solution in 3D
    '''
    
    x = dic[container.mesh_name].source
    
    V = container.V
    mesh_obj = container.mesh_obj
    kappa = container.kappa
    
    y  = container.mesh_obj.coordinates()

    x_y = x-y
    ra  = x_y * x_y
    ra  = np.sum( ra, axis = 1 )
    ra  = np.sqrt( ra ) + 1e-13
    kappara = kappa * ra

    phi_arr = container.factor * np.power( kappara, 0.5 ) * sp.kv( 0.5, kappara )
    phi     = Function( V )
    phi.vector().set_local( phi_arr[dof_to_vertex_map(V)] )

    helper.save_plots( phi, 
                       ["Free Space", "Greens Function"], 
                       container )


def fundamental( container ):
    ''' 
    pretty self explanatory.
    '''
    if container.dim == 2:
        fundamental2D( container )
    if container.dim == 3:
        fundamental3D( container )


def green( container, BC ):
    '''
    plots the domain Green's function with boundary
    condition BC on the mesh for a source.
    '''
    u = container.u
    v = container.v
    normal = container.normal
    kappa = container.kappa
    f = Constant( 0.0 )
    tmp = Function( container.V )

    loc_solver = container.solvers( BC )
    L = f*v*dx
    b = assemble(L)
    
    # Make the RHS have delta funcitons at the sources
    helper.apply_sources( container, b )

    sol = Function( container.V )
    loc_solver( tmp.vector(), b )
    loc_solver( sol.vector(), assemble(tmp*v*dx) )
    
    helper.save_plots( sol,
                       [ BC, "Greens Function", container.quad],
                       container )
    

def variance( container, BC ):
    '''
    plots domain pointwise variance with boundary condition BC.
    '''
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

  

if __name__ == "__main__":

    mesh_name = sys.argv[1]
    if len( sys.argv ) > 2 :
        quad = sys.argv[2]
    else:
        quad = "std"
    print
    print mesh_name

    
    container = container.Container( mesh_name,
                                     dic[mesh_name](), # get the mesh
                                     dic[mesh_name].alpha,
                                     gamma = 1,
                                     quad = quad )
    
    print "fundamental"
    start_time = time.time()
    fundamental( container )
    print "Run time: " + str( time.time() - start_time )
    print

    print "neumann"
    start_time = time.time()
    green(container, "neumann" )
    print "Run time: " + str( time.time() - start_time )
    print

    print "dirichlet"
    start_time = time.time()
    green(container, "dirichlet" )
    print "Run time: " + str( time.time() - start_time )
    print

    if not "square" in mesh_name: 
        
        print "roininen" 
        start_time = time.time()
        green(container, "roininen robin" )
        print "Run time: " + str( time.time() - start_time )
        print

        print "ours"
        start_time = time.time()
        green(container, "ours" )
        print "Run time: " + str( time.time() - start_time )
        print

        print "our robin variance"
        start_time = time.time()
        variance( container, "ours" )
        print "Run time: " + str( time.time() - start_time )
        print

        print "neumann variance"
        start_time = time.time()
        variance( container, "neumann" )
        print "Run time: " + str( time.time() - start_time )
        print
