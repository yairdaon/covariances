import numpy as np
from scipy import special as sp
from scipy.linalg import sqrtm as sqrtm
import math

from dolfin import *

import betas2D

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
                  kappa,
                  gamma = 1.0,
                  num_samples = 0,
                  sqrt_M = None ):

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
        self.kappa  = kappa
        self.kappa2 = kappa**2
        self.gamma  = gamma

        # Dimensionality of the finite element space V
        self.n = self.V.dim()
       
        # Square root of mass matrix, in case we need to sample.
        self._sqrt_M = sqrt_M
        self._M = None

        # Various dictionaries, holding the relevant data for
        # every boundary condition. These are not directly used. 
        # They are called by the properties (functions with 
        # @property above their definition ) below.
        self._variances = {}
        self._gs = {}
        self._form = {}
        self._solvers = {}
  
        # How many samples we'd like to use when approxiamting 
        # pointwise variacne via monte carlo.
        self.num_samples = num_samples
        
        # Calculate the required constants. See below.
        self.set_constants()

    
    def set_constants( self ):
        '''
        We assume the user gives us an the parameters
        gamma and kappa such that the covariance is
        ( - gamma * Laplacian  +  kappa^2  )^{-2}.
        Then we modify these parameters such that
        
        '''
        # Gamma premultiplies the operator
        # -\Delta + kappa^2
        gamma = self.gamma
        
        # We factor out gamma, so we scale kappa
        # accordingly. Later we compensate
        kappa2 = self.kappa2 / gamma
        
        # Here we compensate - we now have covariance
        # [ gamma * (-Delta + kappa^2 / gamma ) ]^2
        self.sig2 = gamma**2 * (4.0*math.pi)**(-self.dim/2.) * kappa2**(-self.nu)
        self.sig  = math.sqrt( self.sig2 )
        self.factor = self.sig2 * 2**(1-self.nu) / math.gamma( self.nu )
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
            gamma = self.gamma
            kappa2 = self.kappa2
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
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx 
                A, _ = assemble_system ( a, f*v*dx, bc )
            
            # Homogeneous Neumann    
            elif "neumann" in BC:
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx
                A = assemble( a )
           
            # Homogeneous Robin using "variant I". Only available in 2D.
            elif "improper" in BC:
                if self.dim == 2:
                    self.imp_beta = betas2D.Beta( self, "imp" )  
                    a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + Max( inner( self.imp_beta, normal ), Constant( 0.0 ) )*u*v*ds
                    A = assemble( a )
                elif self.dim == 3:
                    raise ValueError( "Improper Robin Boundary not supported in 3D. Go home." )
            # Homogeneous Robin using "variant II". Available in both 2D and 3D.    
            elif "mixed" in BC:
                if self.dim == 2:
                    self.mix_beta = betas2D.Beta( self, "mix" )
                elif self.dim == 3:
                    self.mix_beta = betas2D.Mix3D( self, "mix2" )
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx + Max( inner( self.mix_beta, normal ), Constant( 0.0 ) )*u*v*ds
                A = assemble( a )
                
            # The homogeneous Robin BC suggested by Roininen et al.
            elif "naive" in BC:
                a = gamma*inner(grad(u), grad(v))*dx + kappa2*u*v*dx +  1./1.42*kappa*u*v*ds
                A = assemble( a )
                
            else:
                raise ValueError( "Boundary condition type: " + BC + " is not supported. Go home." )
                
            # Keep track of the forms / matrices we have just assembled.
            self._form[BC] = A
        return self._form[BC]

    def gs( self, BC ):
        '''
        The function g used for normalizing the pointwise
        variance
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

        var   = Function( self.V )
        g     = Function( self.V )
        n     = self.n
        loc_solver = self.solvers(BC)
    
        # If we use sampling...
        if self.num_samples > 0:
            tmp   = Function( self.V )
            noise = Function( self.V )
        
            for i in range( self.num_samples ):
        
                # The finite element equivalent of white noise - iid Gaussians weighted by mass matrix.
                # Note that "self.sqrt_M" is a *property* so the first time it is called, the calculation
                # of the square root of the mass matrix is carried out. This may take a while, especially
                # for big meshes.
                noise.vector().set_local( np.einsum( "ij, j -> i", self.sqrt_M, np.random.normal( size = n ) ) )
            
                # Apply inverse operator to white noise.
                loc_solver( tmp.vector(), noise.vector() )

                # Variance is averaged sum of squares.
                var.vector().set_local( var.vector().array() + tmp.vector().array()*tmp.vector().array() )

            # Divide by number of samples to get average sum of squares - this is a maximum likelihood
            # estimator for the variance so it is biased. But just by a tiny tiny bit (if you took the
            # number of samples to be large enough ).
            var.vector().set_local( var.vector().array() / self.num_samples )
    
        # If we choose not to use samples to estimate pointwise variance, we calculate
        # the green's function at the diagonal points (x,x).
        else:
        
            V = self.V
            mesh_obj = self.mesh_obj
            tmp1 = Function( V )
            tmp2 = Function( V )

            b    = assemble( Constant(0.0) * self.v * dx )
                     
            # Get all coordiantes
            coor = V.tabulate_dof_coordinates()
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
        
            # Set the pointwise variacne value we just calculated in the right spot
            var.vector().set_local( vertex_values[dof_to_vertex_map(V)] )

            # Get the function g which is used to normalize the variance
            g.vector().set_local( 
                np.sqrt( self.sig2 / var.vector().array() )
            )
        
        # Keep this data in the place it belongs - in a dictionary.
        self._variances[BC] = var
        self._gs[BC] =  g 
