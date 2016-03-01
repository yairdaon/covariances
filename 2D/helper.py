import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math

pts ={}
pts["square"]           = [ np.array( [ 0.001 , 0.5   ] ) ]
pts["dolfin_coarse"]    = [ np.array( [ 0.45  , 0.65  ] ) ] 
pts["dolfin_fine"]      = [ np.array( [ 0.45  , 0.65  ] ) ] 
pts["pinch"]            = [ np.array( [ 0.35  , 0.155 ] ) ]
pts["l_shape"]          = [ np.array( [ 0.45  , 0.65  ] ),
                            np.array( [ 0.995 , 0.2   ] ),
                            np.array( [ 0.05  , 0.005 ] ) ]

no_scaling =  lambda x: 1.0
def apply_sources (mesh_name, container, b, scaling = no_scaling ):
    sources = pts[mesh_name]
    for source in sources:
        PointSource( container.V, Point ( source ), 1./ scaling(source)  ).apply( b )
        
def get_var_and_g( A, container, k ):
    
    n     = container.n
    tmp   = Function( container.V )
    noise = Function( container.V )
    var   = Function( container.V )
    g     = Function( container.V )

    for i in range(k):
    
        noise.vector().set_local( np.einsum( "ij, j -> i", container.sqrt_M, np.random.normal( size = n ) ) )

        solve( A, tmp.vector(), noise.vector() )
        var.vector().set_local( var.vector().array() + tmp.vector().array()*tmp.vector().array() )
    
    var.vector().set_local( var.vector().array() / k )
    

    g.vector().set_local( 
        np.power( 
            var.vector().array() / container.sig2,
            -0.5 
        )
    )

    return var , g

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
   
    File( "data/" + file_name + ".pvd") << data
