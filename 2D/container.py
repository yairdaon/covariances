import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
from dolfin import *
import pdb
import math

import helper
import betas

class Container():

    def __init__( self,
                  mesh_name,
                  mesh_obj,
                  kappa,
                  gamma = 1.0,
                  num_samples = 0,
                  sqrt_M = None,
                  M = None ):

        self.mesh_name = mesh_name
        self.dim = mesh_obj.geometry().dim()
        self.mesh_obj = mesh_obj
        self.normal = FacetNormal( mesh_obj )
        self.V  =       FunctionSpace( mesh_obj, "CG", 1 )
        self.V2 = VectorFunctionSpace( mesh_obj, "CG", 1 )
       
        # Not sure if need this...
        self.R  = VectorFunctionSpace( mesh_obj, 'R' , 0 )
        self.c  = TestFunction( self.R )

        self.u = TrialFunction( self.V )
        self.v = TestFunction( self.V )
        self.kappa  = kappa
        self.kappa2 = kappa**2
        self.gamma  = gamma
        self.n = self.V.dim()
       
        self._sqrt_M = sqrt_M
        self._M = M
        self._variances = {}
        self._gs = {}
        self._form = {}
  
        self.num_samples = num_samples
        
        self.set_constants()

    def generate( self, name ):
        file = open( "cpp/" + name + ".cpp" , 'r' )  
        code = file.read()
        
        if "enum" in name:
            xp = Expression( code, element = self.V2.ufl_element() )
        else:
            xp = Expression( code, element = self.V.ufl_element() )
            
        xp.kappa  = self.kappa / math.sqrt( self.gamma )
        xp.factor = self.factor
        return xp

    
    def set_constants( self ):
                
        gamma = self.gamma
        
        # We factor out gamma, so we scale kappa
        # accordingly. Later we compensate
        kappa2 = self.kappa2 / gamma
        
        # Here we compensate - we now have covariance
        # [ gamma * (-Delta + kappa^2 / gamma ) ]^2
        self.sig2 = gamma**2 / 4.0 / math.pi / kappa2 
        self.sig  = math.sqrt( self.sig2 )
        self.factor = self.sig2
        
        self.ran_var = ( 0.0, 4 * self.sig2 )
        self.ran_sol = ( 0.0, 2 * self.sig2 )

    @property
    def sqrt_M( self ):
        if self._sqrt_M == None:
            print "Preparing square root of mass matrix. This will take some time."
            self._sqrt_M =  sqrtm( self.M )
            print "Done!"
        return self._sqrt_M

    @property
    def M( self ):
        if self._M == None:
            self._M = assemble( self.u*self.v*dx )
        return self._M

    def form( self, BC ):
       
        if BC in self._form:
            return self._form[BC]
        else:

            gamma = self.gamma
            kappa2 = self.kappa2
            kappa = self.kappa
                       
            u = self.u
            v = self.v
            normal = self.normal
             
            if "dirichlet" in BC:
                def boundary(x, on_boundary):
                    return on_boundary
                f = Constant( 0.0 )
                bc = DirichletBC(self.V, f, boundary)
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx 
                A, _ = assemble_system ( a, f*v*dx, bc )
                
            elif "neumann" in BC:
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx
                A = assemble( a )
                
            elif "mixed" in BC:
                mix_beta = betas.Beta( self, "mix_enum", "mix_denom" )
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + inner( mix_beta, normal )*u*v*ds
                A = assemble( a )
                
            elif "naive" in BC:
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + 1.42*kappa*u*v*ds
                A = assemble( a )
                
            elif "improper" in BC:
                imp_beta = betas.Beta( self, "imp_enum", "imp_denom" )
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + inner( imp_beta, normal )*u*v*ds
                A = assemble(a)
                
            else:
                raise ValueError( "Boundary condition type not supported. Go home." )
                
            self._form[BC] = A
            return self._form[BC]

    def gs( self, BC ):
        if BC in self._gs:
            return self._gs[BC]
        else:
            self._variances[BC], self._gs[BC] = helper.get_var_and_g( self, self.form( BC ) )
            return self._gs[BC]
            
    def variances( self, BC ):
        if BC in self._variances:
            return self._variances[BC]
        else:
            self._variances[BC], self._gs[BC] = helper.get_var_and_g( self, self.form( BC ) )
            return self._variances[BC]
