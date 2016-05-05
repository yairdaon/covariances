import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
from dolfin import *
import pdb
import math

import helper
class Container():

    def __init__( self, mesh_name, mesh_obj, kappa, power, num_samples, sqrt_M = None ):

        self.mesh_name = mesh_name
        self.y  = mesh_obj.coordinates()
        self.dim = mesh_obj.geometry().dim()
        self.mesh_obj = mesh_obj
        self.normal = FacetNormal( mesh_obj )
        self.V  =       FunctionSpace( mesh_obj, "CG", 1 )

        self.u = TrialFunction( self.V )
        self.v = TestFunction( self.V )
        self.kappa = kappa
        self.kappa2 = kappa * kappa
        self.n = self.V.dim()
       
        self._sqrt_M = sqrt_M
        self._variances = {}
        self._gs = {}

        self.num_samples = num_samples
        
        self.power = power
        self.nu = self.power - self.dim / 2.0
        
        nu = self.nu
        dim = self.dim
        kappa = self.kappa
        
        self.sig2 = math.gamma( nu ) / math.gamma( nu + dim/2.0 ) / (4*math.pi)**(dim/2.0) / kappa**(2.0*nu) 
        self.sig  = math.sqrt( self.sig2 )
        self.factor = self.sig2 / 2**(nu-1) / math.gamma( nu )
        
        self.ran_var = ( 0.0, 4 * self.sig2 )
        self.ran_sol = ( 0.0, 2 * self.sig2 )

    @property
    def sqrt_M( self ):
        if self._sqrt_M == None:
            print "Preparing square root of mass matrix. This will take some time."
            self._sqrt_M =  sqrtm( assemble( self.u*self.v*dx ).array() )
            print "Done!"
        return self._sqrt_M
    
    def gs( self, BC ):
        if BC in self._gs:
            return self._gs[BC]
        else:
            helper.set_vg( self, BC )
            return self._gs[BC]
            
    def variances( self, BC ):
        if BC in self._variances:
            return self._variances[BC]
        else:
            helper.set_vg( self, BC )
            return self._variances[BC]
        

class MixedRobin(Expression):

    def __init__( self, container ):
        
        self.container = container
        self.dic = {}
    
    def eval(self, value, x ):
        
        if self.dic.has_key( (x[0], x[1], x[2] ) ):
            t =  self.dic[ ( x[0], x[1], x[2] ) ]
            value[0] = t[0]
            value[1] = t[1]
            value[2] = t[2]
            
        else:
           
            V = self.container.V
            kappa = self.container.kappa
            nu = self.container.nu 
            y  = self.container.y

            x_y = x-y
            ra  = x_y * x_y
            ra  = np.sum( ra, axis = 1 )
            ra  = np.sqrt( ra ) + 1e-13
            kappara = kappa * ra
            
            # Get the denominator
            denom_left_arr = 2.0 * sp.kv( 0.5, kappara )
            denom_left     = Function( V )
            denom_left.vector().set_local( denom_left_arr[dof_to_vertex_map(V)] )
           
            denom_right_arr = np.exp( -kappara ) / np.sqrt( kappara )
            denom_right     = Function( V )
            denom_right.vector().set_local( denom_right_arr[dof_to_vertex_map(V)] )

            denom = assemble( denom_left * denom_right * dx ) 

            # First, second and third components of enumerator
            enum_left_tmp  = kappa * kappa * sp.kv( 0.5, kappara ) * ( 2.0 * kappara + 1.0 )  
            enum_right_tmp = np.exp( -kappara ) * np.power( kappara , -2.5 )
           
            enum_left_arr0 = enum_left_tmp * (x_y)[:,0]
            enum_left0     = Function( V )
            enum_left0.vector().set_local( enum_left_arr0[dof_to_vertex_map(V)] )
            enum_right_arr0 = enum_right_tmp 
            enum_right0     = Function( V )
            enum_right0.vector().set_local( enum_right_arr0[dof_to_vertex_map(V)] )
            enum0 = assemble( enum_left0 * enum_right0 * dx )
            
            enum_left_arr1 = enum_left_tmp * (x_y)[:,1]
            enum_left1     = Function( V )
            enum_left1.vector().set_local( enum_left_arr1[dof_to_vertex_map(V)] )
            enum_right_arr1 = enum_right_tmp 
            enum_right1     = Function( V )
            enum_right1.vector().set_local( enum_right_arr1[dof_to_vertex_map(V)] )
            enum1 = assemble( enum_left1 * enum_right1 * dx )
            
            enum_left_arr2 = enum_left_tmp * (x_y)[:,2]
            enum_left2     = Function( V )
            enum_left2.vector().set_local( enum_left_arr2[dof_to_vertex_map(V)] )
            enum_right_arr2 = enum_right_tmp 
            enum_right2     = Function( V )
            enum_right2.vector().set_local( enum_right_arr2[dof_to_vertex_map(V)] )
            enum2 = assemble( enum_left2 * enum_right2 * dx )
                                  
            value[0] = enum0/denom
            value[1] = enum1/denom
            value[2] = enum2/denom
                                            
            self.dic[(x[0], x[1], x[2])] = (value[0], value[1], value[2]) 
     
    def value_shape(self):
        return (3,)
