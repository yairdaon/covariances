import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math

def apply_sources( file_name, container, b ):
    
    # Modify the right hand side vector to account for point source(s)
    if file_name == "dolfin_coarse.xml" or file_name == "dolfin_fine.xml":
        PointSource( container.V, Point ( 0.45 , 0.65  ), 1. ).apply( b )
    
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
