import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
import math

from dolfin import *

import betas
import radial

class Container():
    '''
    Holds all parameters, ufl forms, solvers
    and pretty much all other data. Does the
    calculation of actually solving the linear
    systems that arise.
    '''
    def __init__( self,
                  mesh_name,
                  mesh_obj,
                  alpha,
                  gamma = 1.0,
                  quad = "std" ):

        # The name of the mesh.
        if "antarctica" in mesh_name:
            mesh_name = "antarctica"
        self.mesh_name = mesh_name

        # The dimensionality of the space - 2 or 3
        self.dim = mesh_obj.geometry().dim()

        # Holds the mesh
        self.mesh_obj = mesh_obj
        
        if self.dim == 3:
            self.x = mesh_obj.coordinates()
            self.nu = 0.5
        elif self.dim == 2:
            self.nu = 1
            
        # The vector space of finite element fucntions that approximates 
        # all of the continuous problems we (try to) solve.
        self.V  =       FunctionSpace( mesh_obj, "CG", 1 )
       
        # The object that represent normal to the boundary
        self.normal = FacetNormal( mesh_obj )

        # Elements of V. Needed to specify finite element
        # ufl form.
        self.u = TrialFunction( self.V )
        self.v = TestFunction( self.V )

        # Parameters we use.
        self.alpha  = alpha
        self.gamma  = gamma
       
        # Various dictionaries, holding the relevant data for
        # every boundary condition. These are not directly used. 
        # They are called by the properties (functions with 
        # @property above their definition ) below.
        self._variances = {}
        self._gs = {}
        self._form = {}
        self._solvers = {}
  
        # Calculate the required constants. See below.
        self.set_constants()

        self.quad = quad
    
    def set_constants( self ):
        '''
        We assume the user gives us an the parameters
        gamma and kappa such that the covariance is
        ( - gamma * Laplacian  +  kappa^2  )^{-2}.
        Then we modify these parameters such that
        
        '''
        
        self.kappa = math.sqrt( self.alpha / self.gamma )
        assert np.isreal( self.kappa )

        # We factor out gamma, so we scale kappa
        # accordingly. Later we compensate
       
        
        # Here we compensate - we now have covariance
        # [ gamma * (-Delta + kappa^2 / gamma ) ]^2
        self.sig2 = ( 
            math.gamma( self.nu )      / 
            math.gamma( 2 )            /
            (4*math.pi)**(self.dim/2.) /
            self.alpha**( self.nu )    /
            self.gamma**( self.dim/2.)
            )
        
        self.sig  = math.sqrt( self.sig2 )
        self.factor = self.sig2 * 2**(1-self.nu) / math.gamma( self.nu )
        self.ran = ( 0.0, 1.3 * self.sig2 )
               
    def solvers( self, BC ):
        '''
        Returns a solver corresponding to the 
        boundary condition described by the 
        string BC. Makes sure we reuse
        the LU factorization as well.
         '''
        
        # If the corresponding solver was not made yet, make it.
        if not BC in self._solvers:
            
            # Use LU solver since this allows reusing the
            # factorization and gain speed up. Here, we call
            # the "form" routine that assembles the matrix used.
            # See documentation of "form" below.
            loc_solver = LUSolver( self.form(BC) )
            loc_solver.parameters['reuse_factorization'] = True 
            
            # This is a solver that knows all it needs to know!!
            # Just give it an input coefficient vector and
            # watch the magic happen.
            self._solvers[BC] = loc_solver.solve
        return self._solvers[BC]

    def form( self, BC ):
        '''
        Returns a matrix A corresponding to the 
        boundary condition described by the 
        string BC. Makes sure we reuse
        the LU factorization as well.
        '''
        
        if not BC in self._form:
            
            # Create local aliases for ease of
            # notation.
            alpha = self.alpha
            gamma = self.gamma
            kappa = self.kappa
            
            u = self.u
            v = self.v
            normal = self.normal
             
            # Go over all possible (and impossible) boundary
            # conditions BC through an if, elif, else statement.
            # Assemble the matrix of the action of:
            # -gamma * Laplacian + kappa^2
            # Note that this isn't the covariance. nor it is the
            # precision. It is a square root of the precision. 
            # For using the covariacne we solve once, assemble the
            # solution and solve again.

            # Homogeneous Dirichlet BC
            if "dirichlet" in BC:
                def boundary(x, on_boundary):
                    return on_boundary
                f = Constant( 0.0 )
                bc = DirichletBC(self.V, f, boundary)
                a = gamma*inner(grad(u), grad(v))*dx + alpha*u*v*dx 
                A, _ = assemble_system ( a, f*v*dx, bc )
            
            # Homogeneous Neumann    
            elif "neumann" in BC:
                a = gamma*inner(grad(u), grad(v))*dx + alpha*u*v*dx
                A = assemble( a )
           
            elif "ours" in BC:
                self.chooseBeta()
                a = (gamma*inner(grad(u), grad(v))*dx + alpha*u*v*dx + 
                     gamma*Max( inner( self.beta, normal ), Constant( 0.0 ) )*u*v*ds
                     )
                A = assemble( a )
                
            # The homogeneous Robin BC suggested by Roininen et al.
            elif "roin" in BC:
                a = gamma*inner(grad(u), grad(v))*dx + alpha*u*v*dx + kappa/1.42*u*v*ds
                A = assemble( a )
                
            else:
                raise ValueError( "Boundary condition type: " + BC + " is not supported. Go home." )
                
            # Keep track of the forms / matrices we have just assembled.
            self._form[BC] = A
        return self._form[BC]

    def chooseBeta(self):
        '''
        Choose the beta we use according the quadrature
        method specified in self.quad variable and dimension
        '''
        if self.dim == 2:
            if "radial" in self.quad:
                self.beta = betas.Beta2DRadial( self ) 
            elif "adaptive" in self.quad:
                self.beta = betas.Beta2DAdaptive( self, tol = 1e-8 )
            elif "std" in self.quad:
                self.beta = betas.Beta2D( self )
            else:
                raise ValueError( "You need to specify the integration method!")
                
        if self.dim == 3:
            if "radial" in self.quad:
                self.beta = betas.Beta3DRadial( self )
            elif "adaptive" in self.quad:
                if "cube" in self.mesh_name:
                    self.beta = betas.BetaCubeAdaptive( self, tol = 1e-8 )
                else:
                    raise ValueError("Adaptive quad may only be used with the cube in 3D.")
            elif "std" in self.quad:
                self.beta = betas.Beta3D( self )
            else:
                raise ValueError( "You need to specify the integration method!")
            
        return self.beta
            
    def gs( self, BC ):
        '''
        The function g used for normalizing the pointwise
        variance. See paper appendix for details.
        '''
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
        '''
        Calculates pointwise variacne and the function g used
        to normalize the variance.
        '''
    
        # Holds pointwise variance values
        var   = Function( self.V )
        
        # Holds sig / sqrt( pointwise variance ). See paper appendix.
        g     = Function( self.V )
        
        # Solves a discretized helmholtz equation with boundary condition BC.
        # this is a square root of the covariance operator.
        loc_solver = self.solvers(BC)
    
        # FE function space
        V = self.V

        # Dimension of FE function space.
        n     = V.dim()

        mesh_obj = self.mesh_obj
        tmp1 = Function( V )
        tmp2 = Function( V )

        # This is going to be a delta function RHS. You'll see.
        b    = assemble( Constant(0.0) * self.v * dx )
                     
        # Get all coordiantes
        coor = V.tabulate_dof_coordinates()
        coor.resize(( V.dim(), self.dim ))
        
        vertex_values = np.zeros(mesh_obj.num_vertices())
            
        # Find pointwise variance for every vertex x as the value
        # of the Green's function at x: G(x,x) = (C delta_x ) (x)
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
        
        # Set the pointwise variacne value we just calculated in the right spot
        var.vector().set_local( vertex_values[dof_to_vertex_map(V)] )

        # Get the function g which is used to normalize the variance
        g.vector().set_local( 
            np.sqrt( self.sig2 / var.vector().array() )
        )
        
        # Keep this data in the place it belongs - in a dictionary.
        self._variances[BC] = var
        self._gs[BC] =  g 
