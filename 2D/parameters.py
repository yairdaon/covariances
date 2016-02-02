import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
from dolfin import *
import pdb
import math

class Container():

    def __init__( self, mesh_obj, kappa, dim, nu ): 

        self.nu = nu
        assert nu % 1 == 0

        self.dim = dim
        self.mesh_obj = mesh_obj
        self.normal = FacetNormal( mesh_obj )
        self.V = FunctionSpace( mesh_obj, "CG", 1 )
        self.u = TrialFunction( self.V )
        self.v = TestFunction( self.V )
        self.kappa = kappa
        self.kappa2 = kappa * kappa
        self.n = self.V.dim()
        self.sqrt_M = sqrtm( assemble( self.u*self.v*dx ).array() )

        # Some constants
        if nu > 0:
            self.sig2 = math.gamma( nu ) / math.gamma( nu + dim/2 ) / (4*math.pi)**(dim/2) / kappa**(2*nu) 
            self.factor = self.sig2 / 2**(nu-1) / math.gamma( nu )
        else:
            self.sig2 = np.Inf
            self.factor = 1.0 / 2.0 / math.pi

    def __call__( self, x, y ):
        '''
        For testing purposes only
        '''
        ra = np.linalg.norm(x-y) + 1e-13 
        return self.factor * (self.kappa*ra)**self.nu * sp.kn(self.nu, self.kappa*ra ) 
        

    def mat( self, x, y ):
        '''
        For testing purposes only
        '''
       
        ret = np.empty( (2,2) )
        ret[0,0] = self(x,y)**2
        ret[0,1] = ret[1,0] = -self(x,y)
        ret[1,1] = 1.0
        
        return ret

    def rhs( self, x, y ):
        '''
        For testing purposes only
        '''
       
        ret  = np.empty( (2,2) )
        ra   = np.linalg.norm(x-y)
        grad = -self.kappa * self.factor * (self.kappa*ra)**(self.nu) * sp.kn( self.nu-1, self.kappa*ra ) * (x-y) / ra 
        ret[0,:] = -self(x,y) * grad
        ret[1,:] = grad
        
        return ret

 

class Robin(Expression):

    def __init__( self, container, param ):
        
        self.container = container
        
        self.mat11 =  self.generate( "mat11" )
        self.mat12 =  self.generate( "mat12" )
        self.mat22 =  self.generate( "mat22" )
                
        self.rhs11 =  self.generate( "rhs11" )
        self.rhs12 =  self.generate( "rhs12" )
        self.rhs21 =  self.generate( "rhs21" )
        self.rhs22 =  self.generate( "rhs22" )

        self.param = param

        self.dic = {}
    
    def eval(self, value, x ):
        
        if self.dic.has_key( (x[0],x[1] ) ):
            t = self.dic[ ( x[0], x[1] ) ]
            value[0] = t[0]
            value[1] = t[1]
        else:
            self.update( x )
        
            # These are the expressions every instance needs
            fe_mat11 = interpolate( self.mat11, self.container.V )   
            mat11 = assemble( fe_mat11 * dx )
                
            fe_rhs11 = interpolate( self.rhs11, self.container.V )
            rhs11 = assemble( fe_rhs11 * dx )
        
            fe_rhs12 = interpolate( self.rhs12, self.container.V )
            rhs12 = assemble( fe_rhs12 * dx )        
        
            if self.param == "hom_beta":
                
                value[0] = rhs11/mat11
                value[1] = rhs12/mat11
                        
            else:

                # These expressions are only required if we use an inhomogeneous
                # Robin boundary condition.
                fe_rhs21 = interpolate( self.rhs21, self.container.V )
                rhs21 = assemble( fe_rhs21 * dx )        
        
                fe_rhs22 = interpolate( self.rhs22, self.container.V )
                rhs22 = assemble( fe_rhs22 * dx )
        
                fe_mat22 = interpolate( self.mat22, self.container.V )
                mat22 = assemble( fe_mat22 * dx )
        
                fe_mat12 = interpolate( self.mat12, self.container.V )
                mat12 = assemble( fe_mat12 * dx )


                det = mat11*mat22 - mat12*mat12
            
                if self.param == "inhom_beta":
                    value[0] = ( mat22*rhs11 - mat12*rhs21 ) / det
                    value[1] = ( mat22*rhs12 - mat12*rhs22 ) / det
        
                elif self.param == "g":
                    value[0] = ( mat11*rhs21 - mat12*rhs11 ) / det  
                    value[1] = ( mat11*rhs22 - mat12*rhs12 ) / det
            
            self.dic[ (x[0],x[1] )] = ( value[0], value[1] )
       
    def update( self, x ):

        self.x = x
        
        self.update_x_xp( x, self.mat11 )
        self.update_x_xp( x, self.mat12 )
        self.update_x_xp( x, self.mat22 )
      
        self.update_x_xp( x, self.rhs11 )
        self.update_x_xp( x, self.rhs12 )
        self.update_x_xp( x, self.rhs21 )
        self.update_x_xp( x, self.rhs22 )
       

    def generate( self, name ):
        file = open( "cpp/" + name + ".cpp" , 'r' )  
        code = file.read()
        xp = Expression( code, element = self.container.V.ufl_element() )
        xp.kappa = self.container.kappa
        xp.factor = self.container.factor
        xp.nu = self.container.nu
        return xp
        
    def update_x_xp( self, x, xp ):
        xp.x[0] = x[0]
        xp.x[1] = x[1]

    def value_shape(self):
        return (2,)

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
           
