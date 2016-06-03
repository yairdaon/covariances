import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
from dolfin import *
import pdb
import math

import mixed3D
import betas2D

class Container():

    def __init__( self,
                  mesh_name,
                  mesh_obj,
                  kappa,
                  gamma = 1.0,
                  num_samples = 0,
                  sqrt_M = None ):

        self.mesh_name = mesh_name
        self.dim = mesh_obj.geometry().dim()
        self.mesh_obj = mesh_obj
        if self.dim == 3:
            self.x = mesh_obj.coordinates()
        elif self.dim == 2:
            self.R  = VectorFunctionSpace( mesh_obj, 'R' , 0 )
            self.c  = TestFunction( self.R )

        self.V  =       FunctionSpace( mesh_obj, "CG", 1 )
        self.V2 = VectorFunctionSpace( mesh_obj, "CG", 1 )
        self.normal = FacetNormal( mesh_obj )
        
        self.u = TrialFunction( self.V )
        self.v = TestFunction( self.V )
        self.kappa  = kappa
        self.kappa2 = kappa**2
        self.gamma  = gamma
        self.n = self.V.dim()
       
        self._sqrt_M = sqrt_M
        self._M = None
        self._variances = {}
        self._gs = {}
        self._form = {}
        self._solvers = {}
  
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
        
        self.ran = ( 0.0, 1.3 * self.sig2 )
       
    @property
    def sqrt_M( self ):
        if self._sqrt_M == None:
            print "Preparing square root of mass matrix. This will take some time."
            self._sqrt_M =  sqrtm( self.M.array() ).real
            print "Done!"
        return self._sqrt_M

    @property
    def M( self ):
        if self._M == None:
            self._M = assemble( self.u*self.v*dx )
        return self._M

    def solvers( self, BC ):
        if not BC in self._solvers:
            loc_solver = LUSolver( self.form(BC) )
            loc_solver.parameters['reuse_factorization'] = True 
            self._solvers[BC] = loc_solver.solve
        return self._solvers[BC]


    def form( self, BC ):
       
        if not BC in self._form:
            
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
           
            elif "improper" in BC:
                if self.dim == 2:
                    self.imp_beta = betas2D.Beta( self, "imp_enum", "imp_denom" )  
                    a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + Max( inner( self.imp_beta, normal ), Constant( 0.0 ) )*u*v*ds
                    A = assemble( a )
                elif self.dim == 3:
                    raise ValueError( "Improper Robin Boundary not supported in 3D. Go home." )
                
            elif "mixed" in BC:
                if self.dim == 2:
                    self.mix_beta = betas2D.Beta( self, "mix_enum", "mix_denom" )
                elif self.dim == 3:
                    self.mix_beta = mixed3D.Mixed( self )
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + Max( inner( self.mix_beta, normal ), Constant( 0.0 ) )*u*v*ds
                A = assemble( a )
                
            elif "naive" in BC:
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + 1.42*kappa*u*v*ds
                A = assemble( a )
                                
            else:
                raise ValueError( "Boundary condition type: " + BC + " is not supported. Go home." )
                
            self._form[BC] = A
        return self._form[BC]

    def gs( self, BC ):
        if BC in self._gs:
            return self._gs[BC]
        else:
            self.set_vg( BC )
            return self._gs[BC]
            
    def variances( self, BC ):
        if BC in self._variances:
            return self._variances[BC]
        else:
            self.set_vg( BC )
            return self._variances[BC]


    def set_vg( self, BC ):
    
        var   = Function( self.V )
        g     = Function( self.V )
        n     = self.n
        loc_solver = self.solvers(BC)
    
        if self.num_samples > 0:
            tmp   = Function( self.V )
            noise = Function( self.V )
        
            for i in range( self.num_samples ):
        
                # White noise, weighted by mass matrix.
                noise.vector().set_local( np.einsum( "ij, j -> i", self.sqrt_M, np.random.normal( size = n ) ) )
            
                # Apply inverse operator to white noise.
                loc_solver( tmp.vector(), noise.vector() )

                # Variance is averaged sum of squares.
                var.vector().set_local( var.vector().array() + tmp.vector().array()*tmp.vector().array() )

                # Divid by number of samples to get average sum of squares.
                var.vector().set_local( var.vector().array() / self.num_samples )
    
        else:
        
            V = self.V
            mesh_obj = self.mesh_obj
            tmp1 = Function( V )
            tmp2 = Function( V )

            b    = assemble( Constant(0.0) * self.v * dx )
                     
            coor = V.dofmap().tabulate_all_coordinates(mesh_obj)
            coor.resize(( V.dim(), self.dim ))

            vertex_values = np.zeros(mesh_obj.num_vertices())
            
            for vertex in vertices( mesh_obj ):
                    
                if self.dim == 2:
                    pt = Point( vertex.x(0), vertex.x(1) )
                elif self.dim == 3:
                    pt = Point( vertex.x(0), vertex.x(1), vertex.x(2) )
                else:
                    raise ValueError( "Dimension not supported. Go home." )
                
                # Apply point source
                PointSource( V, pt, 1.0 ).apply( b )

                # Apply inverse laplacian once ...
                loc_solver( tmp1.vector(), b )

                # ... and twice (and reassemble!!!)
                loc_solver( tmp2.vector(), assemble( tmp1 * self.v * dx ) )
                    
                # Place the value where it belongs.
                vertex_values[vertex.index()] = tmp2.vector().array()[vertex_to_dof_map(V)[vertex.index()]]
            
                # Remove point source
                PointSource( V, pt, -1.0 ).apply( b )
        
            #vertex_values = np.maximum( vertex_values, 0.0 )
            var.vector().set_local( vertex_values[dof_to_vertex_map(V)] )

            g.vector().set_local( 
                np.sqrt( self.sig2 / var.vector().array() )
            )

        self._variances[BC] = var
        self._gs[BC] =  g 
