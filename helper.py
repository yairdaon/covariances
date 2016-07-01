import numpy as np
import math

from dolfin import *

'''
This is a helper file. It contains routines
that are somewhat peripheral to the actual 
math done in a run
'''

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
    name = container.mesh_name
    if "antarctica" in name:
        ant = dic["antarctica"]
        PointSource( container.V, 
                     Point  ( ant.source ),
                     scaling( ant.source )
                 ).apply( b )
        PointSource( container.V, 
                     Point  ( ant.center_source ),
                     scaling( ant.center_source )
                 ).apply( b )
        return
    elif "square" in name:
        source = dic["square"].source
    elif "parallelogram" in name:
        source = dic["parallelogram"].source
    elif "cube" in  name:
        source = dic["cube"].source
    
    PointSource( container.V, 
                 Point ( source ),
                 scaling( source )
                 ).apply( b )
    
def get_mesh( mesh_name ):
    '''
    Generate a mesh.
    '''
    
    pts = [ np.array( [ 0.0, 1.0 ] ),
            np.array( [ 1.0, 1.0 ] ),
            np.array( [ 1.0, 0.0 ] ),
            np.array( [ 0.0, 0.0 ] ),
            np.array( [ 0.0, 1.0 ] ),
            np.array( [ 1.0, 1.0 ] ),
            np.array( [ 1.0, 0.0 ] ),
            np.array( [ 0.0, 0.0 ] )]
       
    if "square" in mesh_name:
        file_name = "../PriorCov/data/square/vertices.txt"
        empty_file( file_name )
        for pt in pts:
            add_point( file_name, pt[0], pt[1] )
        return UnitSquareMesh( dic["square"].x, dic["square"].y )
    
    elif "parallelogram" in mesh_name:
        
        par = dic["parallelogram"]
        mesh_obj = UnitSquareMesh( par.x, par.y )

        # The matrix that sends the unit square
        # to the parallelogram.
        A = par.transformation
                
        # Apply matrix A to all points in xy.
        mesh_obj.coordinates()[:] = np.einsum( "ij, kj -> ki", A, mesh_obj.coordinates() )
         
        file_name = "../PriorCov/data/parallelogram/vertices.txt"
        empty_file( file_name )

        # Now we save to a file the vertices of the
        # parallelogram - for plotting purposes.    
        for pt in pts:
            new_pt = np.dot( A, pt )
            add_point( file_name, new_pt[0], new_pt[1] )
         
        return mesh_obj
    
    elif "antarctica" in mesh_name:
        return Mesh( "meshes/antarctica3.xml" )
    
    elif "cube" in mesh_name:
        return UnitCubeMesh( dic["cube"].x, dic["cube"].y, dic["cube"].z )
        
def get_refined_mesh( mesh_name, 
                      nor = 3, 
                      tol = 0.1,
                      factor = 0.5,
                      greens = False,
                      variance = False,
                      betas = False ):
    '''
    Generate and refine a mesh, usually around a source.
    
    nor is number of refinements - how many refinement 
    iterations we take.
    
    tol - the size of the region we refine initially.

    factor - by how much the region shrinks with each 
    iteration refinement.
    '''

    # Start with an unmodified mesh.
    mesh_obj = get_mesh( mesh_name )
    
    if "cube" in mesh_name:
        
        cub = dic["cube"]
        # if False:
        #     x = mesh_obj.coordinates()[:,0]
        #     y = mesh_obj.coordinates()[:,1]
        #     z = mesh_obj.coordinates()[:,2]
        #     x = np.power( x, cub.s )
        #     z = cub.a*z*z*z + cub.b*z*z + cub.c*z
        #     mesh_obj.coordinates()[:] = np.transpose( np.array( [ x, y, z ] ) )
        
        def inside( x, p ):
            face   = variance and near(x[0], 0.0, tol ) 
            cross  = ( greens and 
                       near(x[1], 0.5 , tol ) and 
                       near(x[0], p[0], tol ) and 
                       near(x[1], p[1], tol ) and 
                       near(x[2], p[2], tol )
                       )
            line   = ( betas and 
                       near(x[0], 0.0, tol ) and 
                       near(x[1], 0.5, tol ) 
                       )
            return face or cross or line
       
    elif "parallelogram" in mesh_name:

        # Selecting edges to refine
        def inside( x, p ):
            A = dic["parallelogram"].transformation
            cross    = ( greens and 
                         near(x[0], p[0], tol ) and 
                         near(x[1], p[1], tol ) 
                         )
            boundary = ( betas and 
                         near( x[1] , x[0]*A[1,1]/A[0,1], tol ) 
                         )
            
            return cross or boundary
                
        # Make a denser mesh towards r=a
        mesh_obj.coordinates()[:] = np.power( mesh_obj.coordinates(), dic["parallelogram"].s )

    
    elif "antarctica" in mesh_name:
        
        # Implements the refine or not function
        # and the default is not to refine at all. 
        inside = lambda x, p: False
        
    elif "square" in mesh_name:
        
         def inside( x, p ):
             return ( greens and 
                      near(x[0], p[0], tol ) and 
                      near(x[1], p[1], tol ) 
                      )
    
    
    # Selecting edges to refine
    class AreaToRefine(SubDomain):
        def inside( self, x, on_boundary ):
            return inside( x, self.loc_source )
    dom = AreaToRefine()
    dom.loc_source = dic[mesh_name].source

    # refine!!!
    for i in range(nor): # nor: Number Of Refinements
        edge_markers = EdgeFunction("bool", mesh_obj)
        dom.mark(edge_markers, True)
        adapt(mesh_obj, edge_markers)
        mesh_obj = mesh_obj.child()
        tol = tol * factor
        
    return mesh_obj


    
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
    
    if "square" in container.mesh_name or "parallelogram" in container.mesh_name:

        line_file   = "../PriorCov/data/" + container.mesh_name + "/line.txt"
        source_file = "../PriorCov/data/" + container.mesh_name + "/source.txt"    
        plot_file   = "../PriorCov/data/" + container.mesh_name + "/" + add_desc( desc ) + ".txt"
        empty_file( line_file, source_file, plot_file )

        source = dic[container.mesh_name].source
        add_point( source_file, source[0], source[1] )

        x_range = np.hstack( ( np.arange( -0.1 , 0.05, 0.001 ),
                               np.arange(  0.05, 0.5 , 0.01  ) ) )
        y = [] 
        x = []
                
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
        loc_file = File( "../PriorCov/data/" + container.mesh_name + "/" + add_desc( desc ) + ".pvd" )
        loc_file << data

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

def empty_file( *args ):
    for file_name in args:
        open(file_name, 'w+').close()

        
def add_point( plot_file, *args ):
    '''
    a routine to add a point to a file
    '''
    dat = ""
    for coordinate in args:
        dat = dat + str(coordinate) + "   " 
    dat = dat + "\n"
    open( plot_file, "a").write(dat)


# The declaration ... = lambda: None is only there
# to create an empty object, since lambda fuctions
# are objects in python is this is super clean IMO.

dic = {}

dic["square"] = lambda: get_mesh( "square" )
dic["square"].kappa = 11.0
dic["square"].x = 855
dic["square"].y = 855
dic["square"].source = np.array( [ 0.05    , 0.5   ] ) 



dic["parallelogram"] = lambda: get_refined_mesh( "parallelogram", nor=3, tol=0.7, factor=0.8, greens=True, betas = True )
dic["parallelogram"].kappa = 11.0
dic["parallelogram"].x = 250
dic["parallelogram"].y = 250
dic["parallelogram"].s = 1.0
dic["parallelogram"].transformation = np.array( [ [ 2.5 , 1.  ],
                                                  [ 1.  , 2.5 ] ] )
dic["parallelogram"].source = np.array( [ 0.025   , 0.025 ] ) 



dic["antarctica"] = lambda: get_refined_mesh( "antarctica", nor=0 )
dic["antarctica"].source        = np.array( [ 7e2     , 5e2   ] )
dic["antarctica"].center_source = np.array( [ -1.5e3  , 600.0 ] ) 
dic["antarctica"].delta = 1e-5
dic["antarctica"].kappa = math.sqrt( dic["antarctica"].delta )
dic["antarctica"].gamma = 1.



dic["cube"] = lambda: None
dic["cube"].kappa = 5.
n = 72
dic["cube"].x = n
dic["cube"].y = n
dic["cube"].z = n 
dic["cube"].source    = np.array( [ 0.05  ,  0.5,  0.5] ) 
dic["cube"].equations = np.array( [ [1    ,    1,    1],
                                    [0.125, 0.25,  0.5], 
                                    [0.75 ,    1,    1]
                                    ] )
dic["cube"].slope = 0.3
dic["cube"].coefficients = np.array( [1, 0.5, dic["cube"].slope ] )
dic["cube"].a, dic["cube"].b, dic["cube"].c = np.linalg.solve( dic["cube"].equations, dic["cube"].coefficients )
dic["cube"].s  = 1.

dic["dolfin"] = lambda: None
dic["dolfin"].source = np.array( [ 0.45    , 0.65  ] ) 
