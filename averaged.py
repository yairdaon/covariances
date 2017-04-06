#!/usr/bin/python
from dolfin import *
import time
import sys
import numpy as np
import scipy.special as sp

import helper
import container
from helper import dic as dic
from simulate import green as green
from simulate import variance as variance

def solver( sol, b ):
    
    # Dangerous but I don't want to change all the
    # rest of the code!!!
    global cot
    
    dirichlet_solver = cot.solvers( "dirichlet" )
    dirichlet = Function( cot.V )
    neumann_solver = cot.solvers( "neumann" )
    neumann = Function( cot.V )
    
    dirichlet_solver( dirichlet.vector(), b )
    neumann_solver  ( neumann.vector()  , b )
    sol.set_local( 
        (neumann.vector().array() + dirichlet.vector().array())/2.
    )
                               
if __name__ == "__main__":

    mesh_name = sys.argv[1]
    print mesh_name
        
    cot = container.Container( mesh_name,
                               dic[mesh_name](), # get the mesh
                               dic[mesh_name].alpha,
                               gamma = 1,
                               numSamples = 0 )
   
    cot._solvers["first averaged"] = solver
    
    #########################################################
    
    print "first averaged green's function" 
    start_time = time.time()
    green( cot, "first averaged" )
    print "Run time: " + str( time.time() - start_time )
    print       

    ##########################################################

    print "second averaged green's function" 
    start_time = time.time()
    dirichlet = green( cot, "dirichlet" )
    neumann   = green( cot, "neumann"   )
    sol = Function( cot.V )
    sol.vector().set_local( 
        (dirichlet.vector().array()+neumann.vector().array())/2.
        )
    helper.save_plots( sol,
                       ["Second Averaged","Greens Function"],
                       cot )
    print "Run time: " + str( time.time() - start_time )
    print       

    ###########################################################

    print "first averaged variance" 
    start_time = time.time()
    variance( cot, "first averaged" )
    print "Run time: " + str( time.time() - start_time )
    print       

    ###########################################################

    print "second averaged variance" 
    start_time = time.time()
    dirichlet_var = cot.stds("dirichlet")
    neumann_var   = cot.stds("neumann")
    std = Function( cot.V )
    var = (
        dirichlet_var.vector().array() * dirichlet_var.vector().array() +
        neumann_var  .vector().array() * neumann_var  .vector().array() 
    ) / 2.

    std.vector().set_local( np.sqrt(var) )
    helper.save_plots( std,
                       ["Second Averaged","Standard Deviation"],
                       cot )
    print "Run time: " + str( time.time() - start_time )
    print       


    
