#!/usr/bin/python
from dolfin import *
import math
import numpy as np
from scipy.interpolate import interp1d as interp

def radial( container, h=None ):
    '''
    This class is a substitute to the analytic expression
    for the free space covariance funciton and its derivative.
    We use it because said expressions have singularities
    at the zero, which we don't like. So, we create a 1D
    funciton (because free space green's function is radially
    symmetric). See more details in paper.
    '''    
    
    # Create mesh and define function space
    # Number of discretization points
    
    # At this distance, the correlation is
    # approximately 10^-3
    if "square" in container.mesh_name:
        ran = 20.
    elif "parallelogram" in container.mesh_name:
        ran = 5.
    elif  "antarctica" in container.mesh_name:
        ran = 8e3
    elif "cube" in container.mesh_name:
        ran = 2.

    # Given a range, we should be able to determine
    # N from the mesh parameter
    if h == None:
        h = container.mesh_obj.hmin()
    N = int( ran / h )
        
    mesh_obj = IntervalMesh( N+1, 0, ran )
    V = FunctionSpace( mesh_obj, "CG", 1 ) 
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant( 0.0 ) 

    # The factors dont REALLY matter, since they cancel out
    # when we calculate beta.
    
    d = container.dim

    # Area of sphere in whatever dimension
    areaOfUnitSphere = 2 * math.pi**(d/2.0) / math.gamma(d/2.0)
    if d == 2:
        X = Expression( str(areaOfUnitSphere) + "* x[0]     ", degree=4) 
    elif d == 3:
        X = Expression( str(areaOfUnitSphere) + "* x[0]*x[0]", degree=4)
    else:
        raise ValueError( "Dimension has to be 2 or 3." )
    
    kappa = container.kappa
    a = X * (kappa*kappa*u*v +  inner(grad(u), grad(v))) * dx 
    m = X *              u*v                             * dx 
    L = f *                v                             * dx

    A, b = assemble_system ( a, L )
    M = assemble( m )

    # Get G1 ###########################

    # Impose rhs delta function at origin
    delta = PointSource ( V, Point ( 0.0 ), 1.0  ) 
    delta.apply ( b )

    # Compute solution
    G1_func = Function(V)
    solve ( A, G1_func.vector(), b )

    # Get G2 via solving for G1 #######
    G2_func = Function(V)
    MG1_func = M*G1_func.vector()
    solve ( A, G2_func.vector(), MG1_func )
        
    coo = np.ravel( mesh_obj.coordinates() ) 
    G1 = []
    G2 = []
    for x in coo:
        G1.append( G1_func(x) ) 
        G2.append( G2_func(x) )

    G1 = np.array( G1 )
    G2 = np.array( G2 )
    
    dG1 = np.zeros( len(G1) )
    dG2 = np.zeros( len(G2) )
    
    # Derivative via finite difference, basically
    dG1[0:-1] = ( G1[1:] - G1[0:-1] ) / h  
    dG2[0:-1] = ( G2[1:] - G2[0:-1] ) / h

    # Use a fast interpolation method,
    # see imports at top
    G1  = interp( coo, G1,  kind = 'linear' )
    G2  = interp( coo, G2,  kind = 'linear' )
    dG1 = interp( coo, dG1, kind = 'zero'   )
    dG2 = interp( coo, dG2, kind = 'zero'   )
        
    return G1, dG1, G2, dG2

if __name__ == "__main__":
    '''
    If you run this module as main, you'll get some plots.
    '''
    from scipy.special import kv as kv
    from scipy.special import kn as kn
    from matplotlib import pyplot as plt

    import helper
    import container
    
    x = np.linspace( 0.0, 0.1, num=500, endpoint=True )
  
    d = 2
    if d == 2:
        cot = container.Container( "square",
                                   helper.get_mesh( "square", 100 ), 
                                   25,
                                   gamma = 1 )
        kappa = cot.kappa
        factor = cot.factor
        Phi1x  =  1. / 2 / math.pi * kn( 0, kappa*x )
        dPhi1x = -1. / 2 / math.pi * kappa * kn( 1, kappa*x )
        Phi2x  =  cot.factor * x * kappa * kn( 1, kappa*x )
        dPhi2x = -cot.factor * kappa * kappa*x * kn( 0, kappa*x )
  
    else:
        cot = container.Container( "cube",
                                   helper.get_mesh( "cube", 99 ),
                                   121,
                                   gamma = 1 )
        kappa = cot.kappa
        factor = cot.factor
        Phi1x  =  1. / 4 / math.pi * np.exp( -kappa*x ) / x
        dPhi1x = -1. / 4 / math.pi * np.exp( -kappa*x ) / x / x * (kappa*x+1) 
        Phi2x  =  cot.factor * np.power(x*kappa,0.5) * kv( 0.5, kappa*x )
        dPhi2x = -cot.factor * kappa * np.power(x*kappa,0.5) * kv( 0.5, kappa*x )
  
    
    G1, dG1, G2, dG2 = radial( cot )
  

    G1x = G1( x )    
    dG1x = dG1( x )    
    G2x = G2( x ) 
    dG2x = dG2( x ) 
        
    plt.plot( x, G1x,         color = 'r', label = 'G1'      )
    plt.plot( x, Phi1x,       color = 'b', label = 'Phi1'    )
    plt.plot( x, Phi1x-G1x,   color = 'g', label = 'Phi1-G1x')
    plt.title( str(d) + "D" )
    plt.ylim([-5,5]) 
    plt.legend()
    plt.show()
    
    plt.plot( x, dG1x,          color = 'r', label = 'dG1'       )
    plt.plot( x, dPhi1x,        color = 'b', label = 'dPhi1'     )
    plt.plot( x, dPhi1x-dG1x,   color = 'g', label = 'dPhi1-dG1x')
    plt.title( str(d) + "D" )
    plt.ylim([-15,15]) 
    plt.legend()
    plt.show()
  
    plt.plot( x, G2x,       color = 'r', label = 'G2'      )
    plt.plot( x, Phi2x,     color = 'b', label = 'Phi2'    )
    plt.plot( x, Phi2x-G2x, color = 'g', label = 'Phi2-G2x')
    plt.title( str(d) + "D" )
    plt.legend()
    plt.show()
    
    plt.plot( x, dG2x,        color = 'r', label = 'dG2'       )
    plt.plot( x, dPhi2x,      color = 'b', label = 'dPhi2'     )
    plt.plot( x, dPhi2x-dG2x, color = 'g', label = 'dPhi2-dG2x')
    plt.title( str(d) + "D" )
    plt.legend()
    plt.show()


    
