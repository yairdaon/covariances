import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math
import os

from fenicstools import StructuredGrid

import parameters

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

def refine_cube( m, n, k, show = False, nor = 2 ):

    mesh_obj = UnitCubeMesh( m, n, k )

    # Break point
    p   = Point( pts["cube"][0] )
    tol = 0.1

    # Selecting edges to refine
    class Border(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], p.x(), tol) and near(x[1], p.y(), tol) and near(x[2], p.z(), tol)
        
    Border = Border()

    # refine!!!
    for i in range(nor):
        edge_markers = EdgeFunction("bool", mesh_obj)
        Border.mark(edge_markers, True)
        adapt(mesh_obj, edge_markers)
        mesh_obj = mesh_obj.child()
        
    if show:
        plot(mesh_obj, interactive=True)
    
    return mesh_obj

def set_vg( container, BC ):

    u = container.u
    v = container.v
    kappa2 = container.kappa2
    kappa = container.kappa
    normal = container.normal

    if "mixed_robin" in BC:
        mix_beta = parameters.MixedRobin( container )
        a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx + inner( mix_beta, normal )*u*v*ds
        A = assemble( a )
        
    elif "naive_robin" in BC:
        a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx + 1.42*kappa*u*v*ds
        A = assemble( a )
    
    elif "neumann" in BC:
        a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx
        A = assemble( a )

    elif "dirichlet" in BC:
        def boundary(x, on_boundary):
            return on_boundary
        f = Constant( 0.0 )
        bc = DirichletBC(container.V, f, boundary)
        a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx 
        A, _ = assemble_system ( a, f*v*dx, bc )
    else:
        raise ValueError( "Boundary condition type not supported. Go home." )
        
    container._variances[BC], container._gs[BC] = get_var_and_g( container, A )


def get_var_and_g( container, A ):
    
    var   = Function( container.V )
    g     = Function( container.V )
    n     = container.n
    
    if container.num_samples > 0:
        tmp   = Function( container.V )
        noise = Function( container.V )
        
        for i in range( container.num_samples ):
        
            # White noise, weighted by mass matrix.
            noise.vector().set_local( np.einsum( "ij, j -> i", container.sqrt_M, np.random.normal( size = n ) ) )
            
            # Apply inverse operator to white noise.
            solve( A, tmp.vector(), noise.vector() )

            # Variance is averaged sum of squares.
            var.vector().set_local( var.vector().array() + tmp.vector().array()*tmp.vector().array() )

        # Divid by number of samples to get average sum of squares.
        var.vector().set_local( var.vector().array() / container.num_samples )
    
    else:
        
        V = container.V
        mesh_obj = container.mesh_obj
        tmp1 = Function( V )
        tmp2 = Function( V )

        b    = assemble( Constant(0.0) * container.v * dx )
                     
        coor = V.dofmap().tabulate_all_coordinates(mesh_obj)
        coor.resize(( V.dim(), container.dim ))

        vertex_values = np.zeros(mesh_obj.num_vertices())
            
        for vertex in vertices( mesh_obj ):
            
            pt = Point( vertex.x(0), vertex.x(1) )
         
            # Apply point source
            PointSource( V, pt, 1.0 ).apply( b )

            # Apply inverse laplacian once ...
            solve( A, tmp1.vector(), b )

            # ... and twice (and reassemble!!!)
            solve( A, tmp2.vector(), assemble( tmp1 * container.v * dx ) )
                    
            # Place the value where it belongs.
            vertex_values[vertex.index()] = tmp2.vector().array()[vertex_to_dof_map(V)[vertex.index()]]
            
            # Remove point source
            PointSource( V, pt, -1.0 ).apply( b )
        
        #vertex_values = np.maximum( vertex_values, 0.0 )
        var.vector().set_local( vertex_values[dof_to_vertex_map(V)] )

    g.vector().set_local( 
        np.sqrt( container.sig2 / var.vector().array() )
    )

    return var, g 
    

def save_plots( data, 
                desc,
                container ):

    source = pts[container.mesh_name][0]
    
    origin = [0., 0.5, 0.]           # origin of slice
    vectors = [[1, 0, 0], [0, 0, 1]] # directional tangent directions (scaled in StructuredGrid)
    dL = [1., 1.]                    # extent of slice in both directions
    N  = [150, 150]                  # number of points in each direction
    
    sl = StructuredGrid(container.V, N, origin, vectors, dL)
    sl(data)
    sl.tovtk(0, filename="../../PriorCov/cube_" + desc[0].replace(" ","_") + "_" + desc[1].replace(" ","_") + ".vtk")

        
    
