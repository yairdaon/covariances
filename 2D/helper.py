import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math
from matplotlib import pyplot as plt
import os

pts ={}
pts["square"]           = [ np.array( [ 0.05  , 0.5   ] ) ]
pts["parallelogram"]    = [ np.array( [ 0.025 , 0.025 ] ) ]
pts["dolfin_coarse"]    = [ np.array( [ 0.45  , 0.65  ] ) ] 
pts["dolfin_fine"]      = [ np.array( [ 0.45  , 0.65  ] ) ] 
pts["pinch"]            = [ np.array( [ 0.35  , 0.155 ] ) ]
pts["l_shape"]          = [ np.array( [ 0.45  , 0.65  ] ),
                            np.array( [ 0.995 , 0.2   ] ),
                            np.array( [ 0.05  , 0.005 ] ) ]

color_counter = 0
colors = [ 'g' , 'b' , 'r', 'k', 'c' , 'm', 'y' ]


no_scaling =  lambda x: 1.0
def apply_sources ( container, b, scaling = no_scaling ):
    sources = pts[container.mesh_name]
    for source in sources:
        PointSource(
            container.V, 
            Point ( source ),
            scaling(source)
        ).apply( b )


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
    
    elif container.num_samples == 0:
        
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
                    
        var.vector()[:] = vertex_values[dof_to_vertex_map(V)]

    else:
        raise ValueError( 'num_samples has to be non-negative' )
        
    g.vector().set_local( 
        np.sqrt( container.sig2 / var.vector().array() )
    )

    return var, g 
 
def save_plots( data, 
                title,
                mesh_name,
                mode = "color",
                ran = [],
                scalarbar = False ):
    
    global color_counter
    x_range = np.arange( 0.0, 0.4, 0.005 )
    y = [] 


    if "square" in mesh_name:
        if "Greens Function" in title or "Fundamental" in title:
        
            source = pts["square"][0]
            slope = 0.0
            intercept = source[1] - slope * source[0]
            line = lambda x: (x, slope * x + intercept )
            
            
            for pt in x_range:
                y.append( data( line(pt) ) ) 
        
            plt.plot( x_range, y, colors[color_counter], label = title )
            color_counter = color_counter + 1
        
            
    elif "parallelogram" in mesh_name:
        if "Greens Function" in title or "Fundamental" in title:
        
            source = pts["parallelogram"][0]
            slope = .8
            intercept = source[1] - slope * source[0]
            line = lambda x: (x, slope * x + intercept )

            for pt in x_range:
                y.append( data( line(pt) ) ) 
        
        plt.plot( x_range, y, colors[color_counter], label = title )
        color_counter = color_counter + 1
        
            
    elif "dolfin" in mesh_name:
        
        file_name =  mesh_name + "_" + title.replace( " ", "_" )

        if ran == []:
            plot( data, 
                  title = title,
                  mode = mode,
                  interactive = False,
                  scalarbar = scalarbar,
              ).write_png( "../../PriorCov/" + file_name )

        elif len(ran) == 2:
            plot( data, 
                  title = title,
                  mode = mode,
                  range_min = ran[0],
                  range_max = ran[1],
                  interactive = False,
                  scalarbar = scalarbar,
              ).write_png( "../../PriorCov/" + file_name )
       
        else:
            raise NameError( "Range is not empty, neither it has two entries" )
            
    else:
        plot( data, title = title )

    #print "Maximum of " + title + " = " + str( np.amax( data.vector().array() ) )
       
def update_x_xp( x, xp ):
    xp.x[0] = x[0]
    xp.x[1] = x[1]

def make_2D_parallelogram( m, n, s = 1.6 ):
        
    par = UnitSquareMesh( m,n )

    # First make a denser mesh towards r=a
    x = par.coordinates()[:,0]
    y = par.coordinates()[:,1]

    x = np.power( x, s )
    y = np.power( y, s )
    
    A = np.array( [ [ 2 , 1 ],
                    [ 1 , 2 ] ] )
    #pdb.set_trace()
    xy = np.array([x, y])
    par.coordinates()[:] = np.einsum( "ij, jk -> ki", A, xy )

    return par

