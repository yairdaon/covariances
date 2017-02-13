#!/usr/bin/python
import numpy as np
import math
import time

import dolfin as dol

import container
import helper
import betas
from helper import dic as dic

'''
This module calculates and saves the data required to
plot betas on the boundary, using the methods suggested
in the paper.
'''
print
print "betas"            


def bdryBetas( mesh_name,
               A,
               quad, # method of quadrature
               nn, # unit normal at the boundary
               n_disc ): # number of disc points per edge
    '''
    THE method that does all the work (or, calls others
    to do its dirty work. Calculate and save betas at
    the boundary
    '''
    
    
    # Where we save the data
    beta_file = "data/" + mesh_name + "/beta_" + quad + "_" + str(n_disc) + ".txt"
    
    # Create an empty file at the desired location
    helper.empty_file( beta_file )

    print beta_file
        
    # Generate the mesh on which calculations take
    # place
    mesh_obj = helper.get_mesh( mesh_name, n_disc )

    # See container.py module for details on this
    # fantastic class
    cot = container.Container( mesh_name,
                               mesh_obj,
                               dic[mesh_name].alpha,
                               quad = quad )

    # The right beta is chosen automatically by
    # the container object, using the quad
    # string and the dimensionality of the mesh.
    beta_obj = cot.chooseBeta()
        
    if "adaptive" in quad:
        # s parametrizes the part of the boundary where we plot betas. 
        for s in np.linspace( 0, 1, num=77, endpoint=True):
            
            # v holds the coordinates in R^d on the unit square
            # or cube. Then it is transformed to the cube or the 
            # parallelogram using A. So if it goes from cube to cube, 
            # then A should be the identity. Luckily, it is.
            v = np.zeros( cot.dim )
            v[-1] = s
            if cot.dim == 3:
                v[1] = .5

            # y is the point where we calculate beta
            y = np.dot( A , v )

            # this is the beta
            beta = beta_obj( y )
            helper.add_point( beta_file,
                              s,
                              np.dot( beta, nn ) )
        return
            
    coo = mesh_obj.coordinates()
    
    # h is a parameter that determines whether or not
    # we are close to the boundary.
    if "parallelogram" in mesh_name:
        h = 1e-5
    elif "square" in mesh_name:
        h = 1e-9
    else:
        h = 1./n_disc

    # Keep track of time
    total_time = 0
    counter = 0

    for i in range(coo.shape[0]):
        
        # Again, y is a point where we calculate betas.
        y = coo[i,:]
        
        # We only do the calculation if it is really close
        # to the boundary. To do this, we make sure we're 
        # thinking of unit square or cube.
        inv = np.linalg.solve( A, y )

        # doit determines if we are close enough to the
        # boundary
        doit = ( abs(inv[0]) < h )
        if cot.dim == 3:
            doit = doit and abs(inv[1]-0.5) < h

        if doit:
            
            # counts how many times we calculated beta
            counter = counter + 1
            
            # start measuring time
            start = time.time()

            # calculate required beta
            beta = beta_obj( y ) 
            
            # add the time the calculation took to the 
            # total time of calculations.
            total_time = total_time + time.time() - start
            
            # save...
            helper.add_point( beta_file,
                              inv[-1],
                              np.dot( beta, nn ) )

    # Pretty self explanatory.
    print "Average time per eval = " + str(total_time/counter)
    
# Take successively more refined meshes.
for i in range( 5, 8 ):
    
    n_disc = 2**i
    
    print
    print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
    print "2D plots, i = " + str(i) + ", n = " + str(n_disc)
    print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
    print

    # Do the plotting for the parallelogram:
    
    # A is the matrix that takes the unit square to the paralleogram
    A = dic["parallelogram"].transformation

    # Unit normal at the boundary on which we plot the betas
    nn = np.array( [ -A[1,1], A[0,1] ] ) / math.sqrt( A[0,1]**2 + A[1,1]**2 )
    
    # Plot using the radial approximation to the Green's function
    bdryBetas( "parallelogram", A, "radial", nn, n_disc )
    
    # Plot using analytic Green's function evaluated at center of FE cells.
    bdryBetas( "parallelogram", A, "std", nn, n_disc )
    
    # Plot the adaptive (i.e. exeact) beta at only the coarsest discretization.
    if i == 5:
        bdryBetas( "parallelogram", A, "adaptive", nn, 2 )
     
    
    # Similar to parallelogram, just simpler.
    A = np.identity( 2 )
    nn = np.array( [ -1.0, 0.0 ] )
    bdryBetas( "square", A, "radial", nn, n_disc )
    bdryBetas( "square", A, "std", nn, n_disc )
    if i == 5:
        bdryBetas( "square", A, "adaptive", nn, 2 )
   
# Pretty similar to previous loop, look there for details.
for i in range( 5, 8 ):
    
    n_disc = 2**i
    
    print
    print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
    print "3D plots, i = " + str(i) + ", n = " + str(n_disc)
    print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
    print
        
    A = np.identity( 3 )
    nn = np.array( [ -1.0, 0.0, 0.0 ] )
    bdryBetas( "cube", A, "radial", nn, n_disc )
    bdryBetas( "cube", A, "std", nn, n_disc )
    if i == 6:
        bdryBetas( "cube", A, "adaptive", nn, 2 )
   
