import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math

def apply_sources( file_name, container, b, g = None ):
    
    # Modify the right hand side vector to account for point source(s)
    if file_name == "dolfin_coarse.xml" or file_name == "dolfin_fine.xml":
        p1 = np.array( [ 0.45 , 0.65 ] )
        if g == None:
            PointSource( container.V, Point ( p1 ), 1. ).apply( b )
        else:
            PointSource( container.V, Point ( p1 ), 1.0 / g(p1) ).apply( b )
    
    elif file_name == "pinch.xml":
        PointSource( container.V, Point ( 0.35 , 0.155  ), 1. ).apply( b )    

    elif file_name == "square":
        PointSource( container.V, Point ( 0.45 , 0.65  ), 1. ).apply( b )
        PointSource( container.V, Point ( 0.995, 0.5   ), 1. ).apply( b )
        PointSource( container.V, Point ( 0.05 , 0.005 ), 1. ).apply( b )
    
    elif file_name == "lshape.xml":
        PointSource( container.V, Point ( 0.45 , 0.65  ), 1. ).apply( b )
        PointSource( container.V, Point ( 0.995, 0.2   ), 1. ).apply( b )
        PointSource( container.V, Point ( 0.05 , 0.005 ), 1. ).apply( b )
  

def get_var( A, container, k ):
    
    n     = container.n
    tmp   = Function( container.V )
    noise = Function( container.V )
    var   = Function( container.V )
    for i in range(k):
    
        noise.vector().set_local( np.einsum( "ij, j -> i", container.sqrt_M, np.random.normal( size = n ) ) )

        solve( A, tmp.vector(), noise.vector() )
        var.vector().set_local( var.vector().array() + tmp.vector().array()*tmp.vector().array() )
    
    var.vector().set_local( var.vector().array() / k )
    return var


def get_g( A, container, k ):
    
    n     = container.n
    tmp   = Function( container.V )
    noise = Function( container.V )
    g   = Function( container.V )
    for i in range(k):
    
        noise.vector().set_local( np.einsum( "ij, j -> i", container.sqrt_M, np.random.normal( size = n ) ) )

        solve( A, tmp.vector(), noise.vector() )
        g.vector().set_local( g.vector().array() + tmp.vector().array()*tmp.vector().array() )
    
    g.vector().set_local( g.vector().array() / k )
    
    g.vector().set_local( 
        np.power( 
            g.vector().array() / container.sig2,
            -0.5 
        )
    ) 
    
    return g 
