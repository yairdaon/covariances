import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
from dolfin import *
import pdb
import math

import helper
class Container():

    def __init__( self, mesh_name, mesh_obj, kappa, dim, nu, num_samples ): 

        self.mesh_name = mesh_name
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

        self._g = None
        self._neumann_var = None
        self.num_samples = num_samples
        
        # Some constants
        if nu > 0:
            self.sig2 = math.gamma( nu ) / math.gamma( nu + dim/2.0 ) / (4*math.pi)**(dim/2.0) / kappa**(2.0*nu) 
            self.sig  = math.sqrt( self.sig2 )
            self.factor = self.sig2 / 2**(nu-1) / math.gamma( nu )
        else:
            self.sig2 = np.Inf
            self.sig  = np.Inf
            self.factor = 1.0 / 2.0 / math.pi

        self.ran_var = ( 0.0, 1.5 * self.sig2 )
        self.ran_sol = ( 0.0, 1.5 * self.sig2 )
        
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

    
    def generate( self, name ):
        file = open( "cpp/" + name + ".cpp" , 'r' )  
        code = file.read()
        xp = Expression( code, element = self.V.ufl_element() )
        xp.kappa = self.kappa
        xp.factor = self.factor
        xp.nu = self.nu
        return xp

    @property
    def g(self):
        if self._g == None:
            self.set_neumann_var_and_g()
        return self._g

    @property
    def neumann_var( self ):
        if self._neumann_var == None:
            self.set_neumann_var_and_g()
        return self._neumann_var

    def set_neumann_var_and_g( self ):
        a = inner(grad(self.u), grad(self.v))*dx + self.kappa2*self.u*self.v*dx
        A = assemble(a)
        self._neumann_var, self._g = helper.get_var_and_g( self, A )

            


class Robin(Expression):

    def __init__( self, container, param ):
        
        self.container = container
        
        self.mat11 =  self.container.generate( "mat11" )
        # self.mat12 =  self.generate( "mat12" )
        # self.mat22 =  self.generate( "mat22" )
                
        self.rhs11 =  self.container.generate( "rhs11" )
        self.rhs12 =  self.container.generate( "rhs12" )
        # self.rhs21 =  self.generate( "rhs21" )
        # self.rhs22 =  self.generate( "rhs22" )

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
        
            if self.param == "imp_beta":
                
                value[0] = rhs11/mat11
                value[1] = rhs12/mat11
            elif self.param == "hom_beta":
                raise ValueError( "Homogeneous Robin for the squared operator not implemented yet.")
            # else:

            #     # These expressions are only required if we use an inhomogeneous
            #     # Robin boundary condition.
            #     fe_rhs21 = interpolate( self.rhs21, self.container.V )
            #     rhs21 = assemble( fe_rhs21 * dx )        
        
            #     fe_rhs22 = interpolate( self.rhs22, self.container.V )
            #     rhs22 = assemble( fe_rhs22 * dx )
        
            #     fe_mat22 = interpolate( self.mat22, self.container.V )
            #     mat22 = assemble( fe_mat22 * dx )
        
            #     fe_mat12 = interpolate( self.mat12, self.container.V )
            #     mat12 = assemble( fe_mat12 * dx )


            #     det = mat11*mat22 - mat12*mat12
            
            #     if self.param == "inhom_beta":
            #         value[0] = ( mat22*rhs11 - mat12*rhs21 ) / det
            #         value[1] = ( mat22*rhs12 - mat12*rhs22 ) / det
        
            #     elif self.param == "g":
            #         value[0] = ( mat11*rhs21 - mat12*rhs11 ) / det  
            #         value[1] = ( mat11*rhs22 - mat12*rhs12 ) / det
            
            self.dic[ (x[0],x[1] )] = ( value[0], value[1] )
       
    def update( self, x ):

        self.x = x
        
        self.update_x_xp( x, self.mat11 )
        # self.update_x_xp( x, self.mat12 )
        # self.update_x_xp( x, self.mat22 )
      
        self.update_x_xp( x, self.rhs11 )
        self.update_x_xp( x, self.rhs12 )
        # self.update_x_xp( x, self.rhs21 )
        # self.update_x_xp( x, self.rhs22 )
       

    # def generate( self, name ):
    #     file = open( "cpp/" + name + ".cpp" , 'r' )  
    #     code = file.read()
    #     xp = Expression( code, element = self.container.V.ufl_element() )
    #     xp.kappa = self.container.kappa
    #     xp.factor = self.container.factor
    #     xp.nu = self.container.nu
    #     return xp
        
    def update_x_xp( self, x, xp ):
        xp.x[0] = x[0]
        xp.x[1] = x[1]

    def value_shape(self):
        return (2,)
