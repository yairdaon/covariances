#!/usr/bin/python
import numpy as np
import math

from dolfin import *

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

    mesh_obj = helper.get_mesh( mesh_name, dims )

    cot = container.Container( mesh_name,
                               mesh_obj,
                               25, #dic[mesh_name].alpha,
                               quad = quad )

    tmp = Function( cot.V )
    L = Constant(1.0) * cot.v * dx
    b = assemble( L )
    loc_solver = cot.solvers( "ours" )
    loc_solver( tmp.vector(), b ) 
                
    beta_dic = cot.beta_xpr.tupDic 
    full_dic = cot.beta_xpr.fullDic

    beta_file = "../PriorCov/data/" + mesh_name + "/beta_" + quad + "_" + str(dims) + ".txt"
    print beta_file
    helper.empty_file( beta_file )
    
    h = 0.5/dims

    for y in beta_dic:
        inv = np.linalg.solve( A, np.array( y ) )
        doit = ( abs(inv[0]) < h )
        if cot.dim == 3:
            doit = doit and abs(inv[1]-0.5) < h
        if doit:
            x = full_dic[y]
            print "Point = " + str(inv)
            print "Beta  = " + str(beta_dic[y])
            print
            helper.add_point( beta_file,
                              inv[-1],
                              np.dot( beta_dic[y], normal ) )
 
# A = dic["parallelogram"].transformation
# nn = np.array( [ -A[1,1], A[0,1] ] ) / math.sqrt( A[0,1]**2 + A[1,1]**2 )
# bdryBetas( "parallelogram", A, "std", nn, 55 )
# bdryBetas( "parallelogram", A, "std", nn, 88 )
# bdryBetas( "parallelogram", A, "std", nn, 99 )

# A = np.identity( 2 )
# nn = np.array( [ -1.0, 0.0 ] )
# bdryBetas( "square", A, "std", nn, 50  )
# bdryBetas( "square", A, "std", nn, 150 )
# bdryBetas( "square", A, "std", nn, 250 )

A = np.identity( 3 )
nn = np.array( [ -1.0, 0.0, 0.0 ] )
bdryBetas( "cube", A, "std"     , nn, 2  )
bdryBetas( "cube", A, "std"     , nn, 7  )
bdryBetas( "cube", A, "std"     , nn, 17 )
bdryBetas( "cube", A, "std"     , nn, 27 )
bdryBetas( "cube", A, "std"     , nn, 37 )

