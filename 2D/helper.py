import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math
from matplotlib import pyplot as plt
import os

pts ={}
pts["square"]           = [ np.array( [ 0.05  , 0.5   ] ) ]
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
        PointSource( container.V, Point ( source ), scaling(source)  ).apply( b )


def get_var_and_g( container, A ):
    
    n     = container.n
    tmp   = Function( container.V )
    noise = Function( container.V )
    var   = Function( container.V )
    g     = Function( container.V )
    
    for i in range( container.num_samples ):
        
        noise.vector().set_local( np.einsum( "ij, j -> i", container.sqrt_M, np.random.normal( size = n ) ) )
        
        solve( A, tmp.vector(), noise.vector() )
        var.vector().set_local( var.vector().array() + tmp.vector().array()*tmp.vector().array() )
                
    var.vector().set_local( var.vector().array() / container.num_samples )
            

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
    
    if mesh_name == "square":
        if "Greens Function" in title or "Fundamental" in title:
        
            global color_counter
            x = np.arange(0.0, 0.5, 0.01 )
            y = []
            
            for pt in x:
                y.append( data( (pt,0.5) ) ) 
        
            plt.plot( x,y, colors[color_counter], label = title )
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

def save_plots( data, title, mesh_name, mode = "color", ran = [] ):

    file_name =  mesh_name + "_" + title.replace( " ", "_" )
    
    if ran == []:
        plot( data, 
              title = title,
              mode = mode,
          ).write_png( "../../PriorCov/" + file_name )

    elif len(ran) == 2:
        plot( data, 
              title = title,
              mode = mode,
              range_min = ran[0],
              range_max = ran[1],
          ).write_png( "../../PriorCov/" + file_name )
       
    else:
        raise NameError( "Range is not empty, neither it has two entries" )
       
def update_x_xp( x, xp ):
    xp.x[0] = x[0]
    xp.x[1] = x[1]
