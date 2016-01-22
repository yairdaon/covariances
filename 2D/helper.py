import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math

class Container():

    def __init__( self, mesh_obj, kappa, dim,  deg=1 ): 

        nu = 2 - dim / 2.0
        self.nu = nu
        
        self.mesh_obj = mesh_obj
        self.V = FunctionSpace( mesh_obj, "CG", deg )
        self.u = TrialFunction( self.V )
        self.v = TestFunction( self.V )
        self.V4 = TensorFunctionSpace( mesh_obj, "CG", deg, shape = (4,) )
        self.kappa = kappa
        self.kappa2 = kappa * kappa
        self.n = self.V.dim()
        
        # Some constants
        self.sig2 = math.gamma( self.nu ) / (
            math.gamma( nu + dim/2.0) * (4*math.pi)**(dim/2.0) * kappa*(2*nu)
        )
        
        self.factor = self.sig2 / (
            2**(nu-1) * math.gamma( nu )
        )

    def __call__( self, x, y ):
        ra = np.linalg.norm(x-y)
        return self.factor * ( self.kappa * ra )**(self.nu) *  sp.kn(self.nu, self.kappa * ra ) 

class Robin(Expression):

    def __init__( self, container ):
        
        self.container = container
        
        mat_file = open( "mat.cpp" , 'r' )  
        mat_code = mat_file.read()
        self.mat = Expression( mat_code )
        self.mat.kappa = container.kappa
        self.mat.factor = container.factor

        rhs_file = open( "rhs.cpp" , 'r' )  
        rhs_code = rhs_file.read()
        self.rhs = Expression( rhs_code )
        self.rhs.kappa = container.kappa
        self.rhs.factor = container.factor
    
    def value_shape(self):
        return (2,)
    
    def eval(self, value, x ):
        '''
        x is a point on the boundary for which we calculate beta according to:
        
                  \int \nabla_x G(x,y) * G(x,y) dy
                  \Omega
        beta(y) = --------------------------------
    
                  \int G^2(x,y) dy
                  \Omega

        where the normal is outward at y and differentiation is wrt y
        '''
        
        
        self.update_x( x )
        fe_mat  = interpolate( self.mat, self.container.V4 ) 
        fe_rhs  = interpolate( self.rhs, self.container.V4 )
    
        mat  = assemble( fe_mat * dx )
        rhs  = assemble( fe_rhs * dx )
        res = np.linalg.solve( mat, rhs )
        
        value[0] = res[0]
        value[1] = res[1]
        value[2] = res[2]
        value[3] = res[3]
        pdb.set_trace()
    def update_x( self, x ):

        self.x = x

        self.mat.x[0] = x[0]
        self.mat.x[1] = x[1]

        self.rhs.x[0] = x[0]
        self.rhs.x[1] = x[1]


class Kappa():
    
    def __init__( self, container ):
    
        self.container = container
        self.beta = Beta( container )
        self.ptwiseVar = container.sig2
        self.normal = FacetNormal( container.mesh_obj )
        
                                                   
        dof_coo = container.V.dofmap().tabulate_all_coordinates(container.mesh_obj)                      
        #pdb.set_trace()
        dof_coo.resize((self.container.n,2))
        self.dof_coo = dof_coo
        

        self.tmp = Function( container.V )
        self.var = Function( container.V )
        self.sol = Function( container.V )
        self.k2  = Function( container.V )  
        self.k2.vector()[:] = container.kappa2
        

        self.L   = Constant( 0.0 ) * container.v * dx
        self.a   = inner(grad(container.u), grad(container.v)) * dx \
                   + self.k2 * container.u * container.v * dx  \
                   # + inner( self.beta, self.normal ) * container.u * container.v * ds 


    def stop( self, count ):
        
        if count % 500 == 0:
            mx    = np.linalg.norm( self.var.vector() - self.ptwiseVar, ord = np.inf )
            print "Iteration " + str(count) + ". Variance Error = " + str(mx)
            print
            if mx < 1E-4:
                return True
            
        return False
        
    def plot(self, pts):
         
        
        src = assemble( self.L ) 
        for pt in pts:
            PointSource( self.container.V, Point(pt), 1. ).apply( src )

        self.A = assemble( self.a )
        solve(self.A, self.sol.vector(), src, 'lu')
        
        plot( self.sol, interactive = True, title = "Solution" )         
        plot( self.var, interactive = True, title = "Variance" )
        plot( self.k2,  interactive = True, title = "kappa" ) 

    def newton(self, iterations = 100, fast = True):

        # if not fast:
        #     G = np.empty( (self.n,self.n) )
        
        for count in range(1,iterations):

            self.A = assemble(self.a)
           
            if self.stop(count):
                break
       
            for i in range( 0, self.container.n ):
        
                pt = Point( self.dof_coo[i] )
                b  = assemble( self.L )
                PointSource( self.container.V, pt, 1. ).apply( b )
    
                solve(self.A, self.tmp.vector(), b, 'lu')
                self.var.vector()[i] = self.tmp.vector()[i][0]
                # if not fast:
                #     G[i,:] = self.tmp.vector()
            
            if fast:
                self.k2.vector()[:] = self.k2.vector()[:] +\
                                       2 * ( self.var.vector() - self.ptwiseVar ) * np.power( self.var.vector(), -4 )                     
            # else: 
            #     self.k2.vector()[:] = self.k2.vector()[:] + \
            #                           np.linalg.solve( G * np.transpose( G ), self.var.vector() - self.ptwiseVar ) 
           
