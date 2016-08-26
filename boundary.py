#!/usr/bin/python
import numpy as np
import math

from dolfin import *

import container
import helper
import betas
from helper import dic as dic

print
print "Parallelogram beta"            


def bdryBetas( mesh_name, 
               beta_ptr, 
               point_func, 
               quad, 
               pt_list, 
               normal,
               dims=[] ):

    mesh_obj = helper.get_mesh( mesh_name, dims )

    cot = container.Container( mesh_name,
                               mesh_obj,
                               dic[mesh_name].alpha,
                               quad = quad )
    beta = beta_ptr( cot )
    
    beta_file = "../PriorCov/data/" + mesh_name + "/beta_" + quad + "_" + str(dims) + ".txt"
    helper.empty_file( beta_file )
    
    for s in pt_list:
        pt = point_func(s)
        helper.add_point( beta_file,
                          s,
                          np.dot( beta(pt), normal )
        )

pt_list = np.linspace(0,1,100)

A = dic["parallelogram"].transformation
xy = lambda s: ( s*A[0,1] , s*A[1,1] )
nn = np.array( [ -A[1,1], A[0,1] ] ) / math.sqrt( A[0,1]**2 + A[1,1]**2 )
bdryBetas( "parallelogram", betas.Beta2DAdaptive, xy, "adaptive", pt_list, nn, 101  )
bdryBetas( "parallelogram", betas.Beta2D        , xy, "std"     , pt_list, nn, 55 )
bdryBetas( "parallelogram", betas.Beta2D        , xy, "std"     , pt_list, nn, 80 )
bdryBetas( "parallelogram", betas.Beta2D        , xy, "std"     , pt_list, nn, 99 )
bdryBetas( "parallelogram", betas.Beta2D        , xy, "std"     , pt_list, nn, 101 )

xy = lambda s: ( 0.0, s )
nn = np.array( [ -1.0, 0.0 ] )
bdryBetas( "square", betas.Beta2DAdaptive, xy, "adaptive", pt_list, nn, 53  )
bdryBetas( "square", betas.Beta2DRadial  , xy, "std"     , pt_list, nn, 50 )
bdryBetas( "square", betas.Beta2D        , xy, "std"     , pt_list, nn, 51 )
bdryBetas( "square", betas.Beta2DRadial  , xy, "std"     , pt_list, nn, 52 )
bdryBetas( "square", betas.Beta2D        , xy, "std"     , pt_list, nn, 53 )

xyz = lambda s: ( 0.0, 0.5, s )
nn = np.array( [ -1.0, 0.0, 0.0 ] )
bdryBetas( "cube", betas.BetaCubeAdaptive, xyz, "adaptive", pt_list, nn, 33 )
bdryBetas( "cube", betas.Beta3DRadial    , xyz, "std"     , pt_list, nn, 30 )
bdryBetas( "cube", betas.Beta3D          , xyz, "std"     , pt_list, nn, 31 )
bdryBetas( "cube", betas.Beta3DRadial    , xyz, "std"     , pt_list, nn, 32 )
bdryBetas( "cube", betas.Beta3D          , xyz, "std"     , pt_list, nn, 33 )
