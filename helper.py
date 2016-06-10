import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math
import os

import matplotlib.pyplot as plt

'''
This is a helper file. It contains routines
that are somewhat peripheral to the run
'''

# pts is a dictionary of source location
# for various meshes.
pts ={}
pts["square"]           = np.array( [ 0.05    , 0.5   ] ) 
pts["tmp"]              = np.array( [ 0.05    , 0.5   ] ) 
pts["parallelogram"]    = np.array( [ 0.025   , 0.025 ] ) 
pts["dolfin"]           = np.array( [ 0.45    , 0.65  ] ) 
pts["pinch"]            = np.array( [ 0.35    , 0.155 ] ) 
pts["antarctica"]       = np.array( [ -1.5e3  , 600.0 ] )
pts["extra"]            = np.array( [ 7e2     , 5e2   ] )
pts["l_shape"]          = np.array( [ 0.45    , 0.65  ] )
pts["cube"]             = np.array( [ 0.05, 0.5, 0.5  ] ) 

# If we do not scale the input (i.e. we don't use variance
# normalization).
no_scaling =  lambda x: 1.0
def apply_sources ( container, b, scaling = no_scaling ):
    '''
    given an assembled right hand side, we add to it
    a delta function at a location specific to the
    mesh
    '''

    # Add two sources in antarctica - one in the bulk
    # and one in the western part, where boundary is
    # tight.
    if "antarctica" in container.mesh_name:
        PointSource( container.V, 
                     Point ( pts["antarctica"] ),
                     scaling( pts["antarctica"] )
                 ).apply( b )
        PointSource( container.V, 
                     Point ( pts["extra"] ),
                     scaling( pts["extra"] )
                 ).apply( b )
        return
    elif "dolfin" in container.mesh_name:
        source = pts["dolfin"]
    else:
        source = pts[container.mesh_name]
    
    PointSource( container.V, 
                 Point ( source ),
                 scaling( source )
             ).apply( b )

def get_source( mesh_name ):
    if "antarctica" in mesh_name:
        return Point ( pts["antarctica"] )
    else:
        return Point ( pts[mesh_name] )
           
    
def refine( mesh_name,
            nor = 3, # Number Of Refinements
            tol = 0.1, # tolerance
            factor = 0.5,
            show = False ):
    '''
    Generate then refine a mesh, usually around a source.
    
    nor is number of refinements - how many refinement 
    iterations we take.
    
    tol - the size of the region we refine initially.

    factor - by how much the region shrinks with each 
    iteration refinement.
    '''
    if "square" in mesh_name:
        mesh_obj = UnitSquareMesh( 50, 50 )
        p  = Point( pts[mesh_name] )    
    elif "parallelogram" in mesh_name:
        mesh_obj = make_2D_parallelogram( 50, 50, 1.4 )
        p  = Point( pts[mesh_name] )
    elif "antarctica" in mesh_name:
        mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )
        p  = Point( pts["antarctica"] )
    else:
        mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )
        p  = Point( pts[mesh_name] )
    
    

    # Selecting edges to refine
    class AreaToRefine(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], p.x(), tol) and near(x[1], p.y(), tol)
    dom = AreaToRefine()

    # refine nor times
    for i in range(nor):
        edge_markers = EdgeFunction("bool", mesh_obj)
        dom.mark(edge_markers, True)
        adapt(mesh_obj, edge_markers)
        mesh_obj = mesh_obj.child()
        tol = tol * factor # change the size of the region
        
    if show:
        plot(mesh_obj, interactive=True)
    
    return mesh_obj



def refine_cube( m, n, k, 
                 nor = 4, 
                 tol = 0.1,
                 factor = 0.5,
                 show = False,
                 greens = False,
                 variance = False,
                 betas = False,
                 slope = 0.25,
                 s = 1.6):
    '''
    See refine method but this is specific for a cube
    in 3D.
    '''
    mesh_obj = UnitCubeMesh( m, n, k )

    # First make a denser mesh towards r=a
  
    if betas:
        
        A = np.array( [ [1    ,    1,    1],
                        [0.125, 0.25,  0.5], 
                        [0.75 ,    1,    1]
                    ] )
        b = np.array( [1, 0.5, slope ] )
        a, b, c = np.linalg.solve( A, b )
        
        x = mesh_obj.coordinates()[:,0]
        y = mesh_obj.coordinates()[:,1]
        z = mesh_obj.coordinates()[:,2]
        x = np.power( x, s )
        z = a*z*z*z + b*z*z + c*z
        mesh_obj.coordinates()[:] =  np.transpose( np.array( [ x, y, z ] ) )

 
    # Break point
    p   = Point( pts["cube"] )
    
    # Selecting edges to refine
    class AreaToRefine(SubDomain):
        def inside(self, x, on_boundary):
            face   = variance and near(x[0], 0.0, tol ) 
            cross  = greens and near(x[1], 0.5, tol ) and near(x[0], p.x(), tol) and near(x[1], p.y(), tol) and near(x[2], p.z(), tol)
            return face or cross 
            
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


def make_2D_parallelogram( m, n, s = 1.6 ):
    '''
    Generate the parallelogram mesh by
    stretching and squeezing
    '''

    par = UnitSquareMesh( m,n )

    # First make a denser mesh towards r=a
    x = par.coordinates()[:,0]
    y = par.coordinates()[:,1]

    x = np.power( x, s )
    y = np.power( y, s )
    
    # The matrix that sends the unit square
    # to the parallelogram.
    A = np.array( [ [ 2.5 , 1   ],
                    [ 1   , 2.5 ] ] )
    
    xy = np.array([x, y])

    # use einsum to apply matrix A to all points xy.
    par.coordinates()[:] = np.einsum( "ij, jk -> ki", A, xy )

    file_name = "../PriorCov/parallelogram.txt"
    try:
        os.remove( file_name )
    except:
        pass

    # Now we save to a file the vertices of the
    # parallelogram - for plotting purposes.    
    pts = [ np.array( [ 0.0, 1.0 ] ),
            np.array( [ 1.0, 1.0 ] ),
            np.array( [ 1.0, 0.0 ] ),
            np.array( [ 0.0, 0.0 ] )]
    for pt in pts:
        new_pt = np.dot( A, pt )
        add_point( file_name, new_pt[0], new_pt[1] )
            
    return par


def add_point( plot_file, *args ):
    '''
    a routine to add a point to a file
    '''
    dat = ""
    for coordinate in args:
        dat = dat + str(coordinate) + "   " 
    dat = dat + "\n"
    with open( plot_file, "a") as myfile:
        myfile.write(dat)


    
def save_plots( data, 
                desc,
                container,
                mode = "color",
                ran = [None, None],
                scalarbar = False ):
    '''
    a routine to save plots and data for plots, based
    on the description variable desc.
    '''

    try:
        os.remove( plot_file )
        os.remove( line_file )
        os.remove( source_file )
    except:
        pass
        
    if "square" in container.mesh_name or "parallelogram" in container.mesh_name:
        x_range = np.hstack( ( np.arange( -0.1 , 0.05, 0.005 ),
                               np.arange(  0.05, 0.5 , 0.01  ) ) )
        y = [] 
        x = []
    
        source = get_source( container.mesh_name )
        line_file   = "../PriorCov/" + container.mesh_name + "_Line.txt"
        source_file = "../PriorCov/" + container.mesh_name + "_Source.txt" 
    
        if "Greens" in add_desc( desc ):

            add_point( source_file, source[0], source[1] )
            if "square" in container.mesh_name:
                slope = 0.0
            else:
                slope = .6

            intercept = source[1] - slope * source[0]
            line = lambda x: (x, slope * x + intercept )

            for pt in x_range:
                try:
                    add_point( plot_file, pt, data( line(pt) ) )
                    add_point( line_file, pt, line(pt)[1] )
                except:
                    pass
  
    else:        
        file_name =   "../PriorCov/" + container.mesh_name + "/" + add_desc( desc )
        loc_file = File( file_name  + ".pvd" )
        loc_file << data

def cube_normal( y ):
    
    # z
    if abs( y[2] - 0.0 ) < 1e-10:
        return ( 0.0, 0.0, -1.0 )
        
    elif abs( y[2] - 1.0 ) < 1e-10:
        return ( 0.0, 0.0, 1.0 )

    # y
    elif abs( y[1] - 0.0 ) < 1e-10:
        return ( 0.0, -1.0, 0.0 )
        
    elif abs( y[1] - 1.0 ) < 1e-10:
        return ( 0.0, 1.0, 0.0 )
    
    # x
    elif abs( y[0] - 0.0 ) < 1e-10:
        return ( -1.0, 0.0, 0.0 )
            
    elif abs( y[0] - 1.0 ) < 1e-10:
        return ( 1.0, 0.0, 0.0 )
            
    # interior
    else:
        return False


def add_desc( str_list ):
    res = ""
    for p in str_list:
        res = res + "_" + p.title().replace(" ","_")
    return res[1:]

def make_tit( desc ):
    res = ""
    for p in desc:
        res = res + " " + p.title()
        return res
