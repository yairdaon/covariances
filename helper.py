import numpy as np
from math import cos
from math import sin
from math import pi
import os

from dolfin import *

'''
This is a helper file. It contains routines
that are somewhat peripheral to the actual 
math done in a run.
'''


# The following piece of codes makes sure we have the directory
# structure where we want to save our data.
dir_list = [ "data", 
             "data/square",
             "data/parallelogram",
             "data/antarctica",
             "data/cube" ]
for path in dir_list:
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# If we do not scale the input (i.e. we don't use variance
# normalization).
no_scaling =  lambda x: 1.0

def apply_sources ( container, b, scaling = no_scaling ):
    '''
    given an assembled right hand side, we add to it
    a delta function at a location specific to the
    mesh.
    The default scaling is none. If we scale then we
    ensure the variance equals 1.
    '''

    # Add two sources in antarctica - one in the bulk
    # and one in the western part, where boundary is
    # tight.
    name = container.mesh_name
    
    # Use source in antarctica mesh
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

    # Source in the other meshes
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
    
def get_mesh( mesh_name, dims ):
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
        file_name = "data/square/vertices.txt"
        empty_file( file_name )
        for pt in pts:
            add_point( file_name, pt[0], pt[1] )
        return UnitSquareMesh( dims, dims )
    
    elif "parallelogram" in mesh_name:
        
        paral = dic["parallelogram"]
        mesh_obj = UnitSquareMesh( dims, dims )
        
        # The matrix that sends the unit square
        # to the parallelogram.
        A = paral.transformation
                
        # Apply matrix A to all points in xy.
        mesh_obj.coordinates()[:] = np.einsum( "ij, kj -> ki", A, mesh_obj.coordinates() )
         
        file_name = "data/parallelogram/vertices.txt"
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
        return UnitCubeMesh( dims, dims, dims )
       
def save_plots( data, 
                desc,
                cot ):
    '''
    a routine to save plots and data for plots, based
    on the description variable desc.
    
    data is a FE function
    '''
       
    # Directory where we save the data
    location = "data/" + cot.mesh_name

    # In square and parallelogram we show a cross section,
    # so we need to code it. The cross section is defined
    # by the equation slope * x + intercept. 
    if "square" in cot.mesh_name or "parallelogram" in cot.mesh_name:
     
        # Creat all the required files to hold the
        # data we generate.
        line_file   = location + "/line.txt"
        source_file = location + "/source.txt"    
        plot_file   = location + "/" + add_desc( desc ) + ".txt"
        empty_file( line_file, source_file, plot_file )

        # Save the source location to a designated file
        source = dic[cot.mesh_name].source
        add_point( source_file, source[0], source[1] )

        # parametrizes the cross section
        x_range = np.hstack( ( np.arange( -0.1 , 0.05, 0.001 ),
                               np.arange(  0.05, 0.5 , 0.01  ) ) )
        y_data = [] 
        x_real = []

        if "square" in cot.mesh_name:
            slope = 0.0
        else:
            slope = .6

        intercept = source[1] - slope * source[0]
        line = lambda x: (x, slope * x + intercept )

        # For every point in the parametrization, see if it
        # gives a point that is inside the square/parallelogram.
        # If so - save it with the right value.
        for pt in x_range:
            try:
                
                # Evaluate the FE funciton at the cross-section.
                # If it is not in the domain this will throw an
                # exception, which we ignore (below).
                y = data( line(pt) )
                add_point( plot_file, pt, y )
                add_point( line_file, pt, line(pt)[1] )
                y_data.append( y  )
                x_real.append( pt )
                
            # The exception mentioned above is ignored
            except:
                pass
 
    # Saving without cross section is so much easier!!!!
    else:        
        loc_file = File( location + "/" + add_desc( desc ) + ".pvd" )
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

dic["square"] = lambda: get_mesh( "square", 256 )
dic["square"].alpha = 121.0
dic["square"].source = np.array( [ 0.05    , 0.5   ] ) 

dic["parallelogram"] = lambda: get_mesh( "parallelogram", 128 )
dic["parallelogram"].alpha = 121.
dic["parallelogram"].transformation = np.array( [ 
    [ cos(pi/4-pi/8) , cos(pi/4+pi/8)  ],
    [ sin(pi/4-pi/8) , sin(pi/4+pi/8)  ] ] )
dic["parallelogram"].source = np.array( [ 0.025   , 0.025 ] ) 

dic["antarctica"] = lambda: get_mesh( "antarctica", 0 )
dic["antarctica"].source        = np.array( [ 7e2     , 5e2   ] )
dic["antarctica"].center_source = np.array( [ -1.5e3  , 600.0 ] ) 
dic["antarctica"].alpha = 1e-5
dic["antarctica"].gamma = 1.

dic["cube"] = lambda: get_mesh( "cube", 64 )
dic["cube"].alpha = 25.
dic["cube"].source    = np.array( [ 0.05  ,  0.5,  0.5] )
