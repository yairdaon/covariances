import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math
import os

# from fenicstools import StructuredGrid


pts ={}
pts["cube"] = [ np.array( [ 0.05, 0.5, 0.5 ] ) ]

no_scaling =  lambda x: 1.0
def apply_sources ( container, b, scaling = no_scaling ):
    sources = pts[container.mesh_name]
    for source in sources:
        PointSource(
            container.V, 
            Point ( source ),
            scaling(source)
        ).apply( b )

def refine_cube( m, n, k, 
                 nor = 3, 
                 tol = 0.1,
                 factor = 0.5,
                 show = False ):

    mesh_obj = UnitCubeMesh( m, n, k )

    # Break point
    p   = Point( pts["cube"][0] )
    
    # Selecting edges to refine
    class AreaToRefine(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], p.x(), tol) and near(x[1], p.y(), tol) and near(x[2], p.z(), tol)
    dom = AreaToRefine()

    # refine!!!
    for i in range(nor):
        edge_markers = EdgeFunction("bool", mesh_obj)
        dom.mark(edge_markers, True)
        adapt(mesh_obj, edge_markers)
        mesh_obj = mesh_obj.child()
        tol = tol * factor
        
    if show:
        plot(mesh_obj, interactive=True)
    
    return mesh_obj

def save_plots( data, 
                desc,
                container ):

    filename = "../../PriorCov/" + container.mesh_name + "_" + desc[0].replace(" ","_") + "_" + desc[1].replace(" ","_") + ".pvd"
    loc_file = File( filename )
    loc_file << data

    # source = pts[container.mesh_name][0]
    
    # origin = [0., 0.5, 0.]           # origin of slice
    # vectors = [[1, 0, 0], [0, 0, 1]] # directional tangent directions (scaled in StructuredGrid)
    # dL = [1., 1.]                    # extent of slice in both directions
    # N  = [150, 150]                  # number of points in each direction

    # sl = StructuredGrid(container.V, N, origin, vectors, dL)
    # sl(data)
    # sl.tovtk(0, filename = filename )

        
    
