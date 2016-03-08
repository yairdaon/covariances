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

        self._neumann_g = None
        self._neumann_var = None
        self._robin_g = None
        self._robin_var = None
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
        

    def mat11( self, x, y ):
        '''
        For testing purposes only
        '''
       
        return self(x,y)**2
        
    def rhs11( self, x, y ):
        '''
        For testing purposes only
        '''
       
        ra   = np.linalg.norm(x-y)
        grad = -self.kappa * self.factor * (self.kappa*ra)**(self.nu) * sp.kn( self.nu-1, self.kappa*ra ) * (x-y) / ra 
        return (-self(x,y) * grad)[0] 
        
    def rhs12( self, x, y ):
        '''
        For testing purposes only
        '''
       
        ra   = np.linalg.norm(x-y)
        grad = -self.kappa * self.factor * (self.kappa*ra)**(self.nu) * sp.kn( self.nu-1, self.kappa*ra ) * (x-y) / ra 
        return (-self(x,y) * grad)[1] 
        

    def generate( self, name ):
        file = open( "cpp/" + name + ".cpp" , 'r' )  
        code = file.read()
        xp = Expression( code, element = self.V.ufl_element() )
        xp.kappa = self.kappa
        xp.factor = self.factor
        xp.nu = self.nu
        return xp

    @property
    def neumann_g(self):
        if self._neumann_g == None:
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

    @property
    def robin_g(self):
        if self._robin_g == None:
            self.set_robin_var_and_g()
        return self._g

    @property
    def robin_var( self ):
        if self._robin_var == None:
            self.set_robin_var_and_g()
        return self._robin_var

    def set_robin_var_and_g( self ):
        a = inner(grad(self.u), grad(self.v))*dx + self.kappa2*self.u*self.v*dx + self.kappa*self.u*self.v*ds
        A = assemble(a)
        self._robin_var, self._g = helper.get_var_and_g( self, A )

            


class Robin(Expression):

    def __init__( self, container, param ):
        
        self.container = container
        
        self.mat11 =  self.container.generate( "mat11" )
                    
        self.rhs11 =  self.container.generate( "rhs11" )
        self.rhs12 =  self.container.generate( "rhs12" )
    
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
                    
            self.dic[ (x[0],x[1] )] = ( value[0], value[1] )
       
    def update( self, x ):

        self.x = x
        
        helper.update_x_xp( x, self.mat11 )
              
        helper.update_x_xp( x, self.rhs11 )
        helper.update_x_xp( x, self.rhs12 )
                
   
    def value_shape(self):
        return (2,)
