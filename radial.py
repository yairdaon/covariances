#!/usr/bin/python
from dolfin import *
import math
import numpy as np

class Radial(Expression):
    
    def __init__( self, container, p ):

        self.container = container
        
        self._degree = 2
        d = self.container.dim
        kappa = container.kappa
               
        # Create mesh and define function space
        # Number of discretization points
        
        # At this distance, the correlation is
        # approximately 10^-3
        ran = max( 10 / container.kappa, 5 )
        if "antarctica" in container.mesh_name:
            ran = 8e3
        # Given a range, we should be able to determine
        # N from the mesh parameter
        N = ran / container.mesh_obj.hmin()
        
        mesh_obj = IntervalMesh(int(N) + 1, 0, int(ran) )
        V = FunctionSpace(mesh_obj, "CG", 1) 
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant( 0.0 ) 

        # The factors dont REALLY matter, since they cancel out
        
        areaOfUnitSphere = 2 * math.pi**(d/2.0) / math.gamma(d/2.0)
        if d == 2:
            X = Expression( str(areaOfUnitSphere) + "* x[0]     ", degree=4) 
        else:
            X = Expression( str(areaOfUnitSphere) + "* x[0]*x[0]", degree=4)
            
        a = X * (kappa*kappa*u*v +  inner(grad(u), grad(v))) * dx 
        m = X *              u*v                             * dx 
        L = f*v*dx

        A, b = assemble_system ( a, L )
        M = assemble( m )

        # Get G1 ###########################

        # Impose rhs delta function at origin
        delta = PointSource ( V, Point ( 0.0 ), 1.0  ) 
        delta.apply ( b )

        # Compute solution
        G1 = Function(V)
        solve ( A, G1.vector(), b )

        if p == 1:
            self.radial = G1
        else:    
            
            # Get G2 via solving for G1 #######
            G2 = Function(V)
            MG1 = M*G1.vector()
            solve ( A, G2.vector(), MG1 )
            self.radial = G2

    def eval( self, value, x ):
        value[0] = self.radial( np.linalg.norm( self.y - x ) )

if __name__ == "__main__":

    import container
    from helper import dic as dic
    
    def err( xpr1, xpr2, space ):  
        return errornorm( 
            interpolate( xpr1, space ),
            interpolate( xpr2, space )
            ) #, norm_type = 'L1' )
  
    cot = container.Container( "square",
                               dic["square"](), 
                               dic["square"].alpha,
                               gamma = 1 )

    kappa = cot.kappa
    factor = cot.factor
    sig2 = cot.sig2

    # Compile Phi1
    phi1_file = open( "cpp/fundamental2d1.cpp" , 'r' )  
    phi1_code = phi1_file.read()
    phi1_file.close()
    phi1 = Expression( phi1_code, kappa=kappa, degree = 2)
    
    #Compile Phi2
    phi2_file = open( "cpp/fundamental2d2.cpp" , 'r' )  
    phi2_code = phi2_file.read()
    phi2_file.close()      
    phi2 = Expression( phi2_code, kappa=kappa, factor = factor, sig2 = sig2 )
    
    
    G1 = Radial( cot, 1 )
    G2 = Radial( cot, 2 )

    print "Norm G2-phi2 = "      + str( err(G2, phi2, cot.V) )
    print "Norm G1-phi1 = "      + str( err(G1, phi1, cot.V) )

