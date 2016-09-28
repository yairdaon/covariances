#!/usr/bin/python
import numpy as np
import math
import time

import dolfin as dol

import container
import helper
import betas
from helper import dic as dic

print
print "betas"            


def bdryBetas( mesh_name,
               A,
               quad,
               normal,
               dims ):
    
    beta_file = "../PriorCov/data/" + cot.mesh_name + "/beta_" + quad + "_" + str(dims) + ".txt"
    helper.empty_file( beta_file )
    print beta_file
        
    mesh_obj = helper.get_mesh( mesh_name, dims )

    cot = container.Container( mesh_name,
                               mesh_obj,
                               dic[mesh_name].alpha,
                               quad = quad )

    beta_obj = cot.chooseBeta()
        
    if "adaptive" in quad:
        
        for s in np.linspace( 0, 1, num=77, endpoint=True):
            v = np.zeros( cot.dim )
            v[-1] = s
            if cot.dim == 3:
                v[1] = .5
            y = np.dot( A , v )
            beta = beta_obj( y )
            helper.add_point( beta_file,
                              s,
                              np.dot( beta, normal ) )
        return
            
    coo = mesh_obj.coordinates()
    

    if "parallelogram" in mesh_name:
        h = 1e-5
    elif "square" in mesh_name:
        h = 1e-9
    else:
        h = 1./dims
    total_time = 0
    counter = 0

    for i in range(coo.shape[0]):
        
        y = coo[i,:]
        inv = np.linalg.solve( A, y )
        doit = ( abs(inv[0]) < h )

        if cot.dim == 3:
            doit = doit and abs(inv[1]-0.5) < h
        if doit:
            counter = counter + 1
            start = time.time()
            beta = beta_obj( y ) 
            total_time = total_time + time.time() - start
            # print "Point = " + str( y )
            # print "Beta  = " + str(beta)
            # print
            
            helper.add_point( beta_file,
                              inv[-1],
                              np.dot( beta, normal ) )
    print "Average time per eval = " + str(total_time/counter)

for i in range( 5, 8 ):
    
    n_disc = 2**i
    
    print
    print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
    print "2D plots, i = " + str(i) + ", n = " + str(n_disc)
    print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
    print

    A = dic["parallelogram"].transformation
    nn = np.array( [ -A[1,1], A[0,1] ] ) / math.sqrt( A[0,1]**2 + A[1,1]**2 )
    bdryBetas( "parallelogram", A, "radial", nn, n_disc )
    bdryBetas( "parallelogram", A, "std", nn, n_disc )
    
    if i == 4:
        bdryBetas( "parallelogram", A, "adaptive", nn, 2 )
     
    
    A = np.identity( 2 )
    nn = np.array( [ -1.0, 0.0 ] )
    bdryBetas( "square", A, "radial", nn, n_disc )
    bdryBetas( "square", A, "std", nn, n_disc )
    if i == 4:
        bdryBetas( "square", A, "adaptive", nn, 2 )
   

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
    if i == 4:
        bdryBetas( "cube", A, "adaptive", nn, 2 )
   
