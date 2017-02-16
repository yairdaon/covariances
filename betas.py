import numpy as np
import hashlib 
import math
from scipy.special import kn as kn
from scipy.special import kv as kv
import time

import dolfin as dol
import instant

import helper
import radial

'''
All classes in this module have the same purpose. They
are used to calculate betas as integrals over domain. This
is done by first calculating every component in the enumerator
(before it is dotted with the outward pointing unit normal) and
the denominator.

The common structure is that every object's __init__ method calls
the generalInit method defined just below. Then, whenever 
FENICS needs to evaluate the expression it calls the object's eval
method. This method is assigned as generalEval. So whenever a
beta value is required - the method generalEval is called. If the
value was not calculated before, the object's strictEval method is
called and that calculates and returns the required value. If the
value was calculated in the past - it is stored in the object's dic
variable. When a value is needed, the generalEval method pulls the
right value from the dic variable.
'''

# #################################################################
# # The free methods below are used since you can't sub-subclass ##
# # a FEniCS Expression class but we still want them used by all ##
# # classes in this moodule.                                     ##
# #################################################################
def generalEval( self, value, y ):
    
    '''
    This is the eval method for the Expression class.
    It just checks whether we already did the 
    calculation it is asked to do. If we did -the
    result is returned. Otherwise, the specific
    strictEval method is called. The strictEval 
    method is implemented for every beta.
    '''

    dim = self.dim

    # Generate the key for the dicitonary ...
    if dim == 2:
        tupKey = (y[0],y[1])
    elif dim == 3:
        tupKey = (y[0],y[1],y[2])

    # ...if the key exists, this means we're already
    # done the calculation, so we return the value stored...
    if self.dic.has_key( tupKey ):
        t = self.dic[ tupKey ]
        for i in range(dim):
            value[i] = t[i]

    # ... If not, we have to do the calculation and store its result.
    else:

        # Do the full calculation
        self.strictEval( value, y )

        # Store the result
        if dim == 2:
            self.dic[ tupKey ] = (value[0], value[1]) 
        if dim == 3:
            self.dic[ tupKey ] = (value[0], value[1], value[2]) 
    
def generalInit( self, cot ):
    '''
    This method initializes any of the Beta classes.
    cot is a container object that holds pretty much
    everything. See container.py for its documentation.
    '''
 
    # This ensures we don't try to evaluate the fundamental
    # solution at vertices since it is singular there.
    # The DOF for this space are in the middle of cells, not 
    # on vertices.
    self.V = dol.FunctionSpace( cot.mesh_obj, "DG", 0 )   
    u = dol.TrialFunction( self.V )
    v = dol.TestFunction ( self.V )
    
    # Mass matrix. Required for approximating integrals of
    # functions on the FE space.
    self.M = dol.assemble( u*v*dol.dx )

    # Hold values of the enumerator components, evaluated
    # at the center of every cell (== DOF of FE function space).
    self.enum0 = dol.Function( self.V )
    self.enum1 = dol.Function( self.V )
    if cot.dim == 3:
        self.enum2 = dol.Function( self.V )
        
    # Same for the denominator
    self.denom = dol.Function( self.V )

    # Get all coordiantes
    self.x = self.V.tabulate_dof_coordinates()
    self.x.resize(( self.V.dim(), cot.dim ))
    
    #See the documentation in container.py
    self.cot = cot 

    # The spatial dimension of the problem (e.g. 2 or 3)
    self.dim = cot.dim

    # Will hold beta values at DOFs. Will be populated by
    # the strictEval method of its class.
    self.dic = {} 
    
def getRaKappara( self, y ):
    '''
    Just a short method cuz we use this all the time
    '''

    # get an array for the differences of the
    # the point queried (i.e. y) and DOFS
    # coordinates (i.e. x).
    y_x = y - self.x
    
    # get corresponding distances
    ra  = np.sqrt( np.sum( y_x * y_x, axis = 1 ) )
    
    # get the kappa * distance
    kappara = self.cot.kappa * ra

    return ra, kappara, y_x
      

#################################################################
# Finite Element betas using FEniCS package #####################
################################################################# 

class Beta2D(dol.Expression):
    '''
    Implementing the optimal beta derived 
    in the paper as a FENICS Expression.
    '''
    def __init__( self, cot ):
        
        # See comment on this method's 
        # definition above.
        generalInit( self , cot )
       
    # This is the method called when we need to return
    # a beta at some boundary point. If we did not 
    # calculate it before, then generalEval calls 
    # the specific strictEval below.
    eval = generalEval

    def strictEval( self, value, y ):
        ''' 
        In order to calculate our optimal betas, integration
        needs to take place. We use the given FE mesh
        Basically, performs Riemann integration with 
        respect to the partition defined by the 
        discontinuous Galerkin finite element discretization.
        The integration point (which in Riemann integration
        can be everywhere) is the DOF of
        '''
                
        ra, kappara, y_x = getRaKappara(self, y )
        
        # These just come from the formulas for Beta. kn 
        # is a scipy method. It is the modified Bessel
        # function of second kind.        
        bess0 = kn( 0, kappara )
        bess1 = kn( 1, kappara )
        enums = self.cot.kappa * ( bess0*bess0 + bess1*bess1 ) 

        enum  = np.einsum( "i , ik -> ik", enums, y_x )
        denom = 2 * ra * bess0 * bess1   
        
        self.enum0.vector().set_local( enum[:,0] )
        self.enum1.vector().set_local( enum[:,1] )
        self.denom.vector().set_local( denom     )

        # Multiply with mass matrix which, in this case, is diagonal and holds
        # cell area/volume on its diagonal (because it is mass matrix of a DG
        # FE space).
        self.enum0.vector().set_local( (self.M * self.enum0.vector()).array() )
        self.enum1.vector().set_local( (self.M * self.enum1.vector()).array() )
        self.denom.vector().set_local( (self.M * self.denom.vector()).array() )
        
        # These sums are just Riemann integration over cells.
        denom = np.sum( self.denom.vector().array() )
        value[0] = np.sum( self.enum0.vector().array() ) / denom
        value[1] = np.sum( self.enum1.vector().array() ) / denom
  
      
    def value_shape(self):
        return (2,)     
       

class Beta3D(dol.Expression):
    '''
    Implementing the optimal beta derived 
    in the paper as a FENICS Expression.
    This class is very similar to the
    Beta2D class. The part that forced
    me to actually create two different
    classes is the value_shape method.
    You need it to be defined (because it
    is called) before __init__ is called.
    '''
    def __init__( self, cot ):
        
        # See comment on this method's 
        # definition above.
        generalInit( self , cot )
       
    # This is the method called when we need to return
    # a beta at some boundary point. If we did not 
    # calculate it before, then generalEval calls 
    # the specific strictEval below.
    eval = generalEval
    
    def strictEval( self, value, y ):
        ''' 
        In order to calculate our optimal betas, integration
        needs to take place. We use the given FE mesh
        Basically, performs Riemann integration with 
        respect to the partition defined by the 
        discontinuous Galerkin finite element discretization.
        The integration point (which in Riemann integration
        can be everywhere) is the DOF of
        '''

        ra, kappara, y_x = getRaKappara(self, y )
        
        # These just come from the formulas for Beta.
        expon = np.exp(-kappara) 
        bess  = kv( 0.5, kappara )
        enums = self.cot.kappa * np.power(ra,-1.5) * (2+1./kappara) * expon * bess 
        enum  = np.einsum( "i , ik -> ik", enums, y_x )
        denom = 2 * bess * expon * np.power(ra,-0.5) 
        
        self.enum0.vector().set_local( enum[:,0] )
        self.enum1.vector().set_local( enum[:,1] )
        self.enum2.vector().set_local( enum[:,2] )
        self.denom.vector().set_local( denom )            
        
        # Multiply with mass matrix which, in this case, is diagonal and holds
        # cell area/volume on its diagonal. 
        self.enum0.vector().set_local( (self.M * self.enum0.vector()).array() )
        self.enum1.vector().set_local( (self.M * self.enum1.vector()).array() )
        self.enum2.vector().set_local( (self.M * self.enum2.vector()).array() )
        self.denom.vector().set_local( (self.M * self.denom.vector()).array() )
        
        # These sums are just Riemann integration over cells.
        denom = np.sum( self.denom.vector().array() )
        value[0] = np.sum( self.enum0.vector().array() ) / denom
        value[1] = np.sum( self.enum1.vector().array() ) / denom
        value[2] = np.sum( self.enum2.vector().array() ) / denom
   
    def value_shape(self):
        return (3,)


#################################################################
# Adaptive betas using the cubature package #####################
#################################################################
class Beta2DAdaptive(dol.Expression):
    '''
    Compile and integrate a given expression. Expression
    is given as a string that refers to a c++ file
    which is then compiled. Uses adaptive quadrature
    from the package cubature. That's is a whole mess on its
    own.
    '''
    
    def __init__( self, cot, tol = 1e-9 ):
        
        # We always need this, because of subclassing
        # issues discussed above.
        generalInit( self , cot )
        
        # Generate the expression from the c++ file that
        # the input variable expresion_string refers to.
        self.xpr = generateInstant( "beta2D" )
            
        self.coordinates = cot.mesh_obj.coordinates()
        self.elements = dict((cell.index(), cell.entities(0)) for cell in dol.cells(cot.mesh_obj))
    
        self.tol = tol
        
    # See comments on this assignment in previous classes.
    # The method generalEval is defined at the start of the
    # file.
    eval = generalEval

    def strictEval( self, value, y ):
        '''
        Here we iterate over FE cells and perform
        exact (i.e. adaptive) quadrature on each cell,
        then sum
        '''
        
        # the dimension
        d = self.dim
    
        # We need to perform d integrals for the 
        # enumerator and one for the denominator.
        result = np.zeros( (d+1,) )

        # Every cell has (d+1) vertices, each has
        # d coordinates. This array stores this data.
        vertices = np.empty(  (d+1)*d  )
     
        # Iterate over cells and perform adaptive quadrature
        # over every cell, then sum them. Uggghhh...
        for cell in self.cot.mesh_obj.cells():
            i = 0
            for vertex in cell:
                vertices[i:i+d] = self.coordinates[vertex]
                i = i + d
            
            # The integrateVector method calculates and adds the result of integration
            # of all components on a cell.
            self.xpr.integrateVector(  y, self.cot.kappa, vertices, self.tol, result )  
        
        # The last entry is the denominator, by which we divide.
        # The first entries are the enumerators, which is a vector
        # that will later be multiplied by a unit normal
        for i in range( d ):
            value[i] = result[i] / result[d]
        
    
    # Should change this to apply to three dimensions!!!
    def value_shape(self):
        return (2,)
    
def generateInstant( expression_string ):
    '''
    Generate the expression from the c++ file that
    the input variable refers to. Tweak at your own
    risk.
    '''
        
    xpr_file = open( "integrate_vector.cpp" , 'r' )  
    xpr_code = xpr_file.read()
    xpr_file.close()
    
    # Replace the function we use into the text of the c++ file.. 
    xpr_code = xpr_code.replace( "FUNCTION_NAME", expression_string )
        
    # Compile and return an object with "integrate vector" method
    xpr = instant.build_module(
        code = xpr_code,
        sources = [ "helper.cpp", "hcubature.c" ],
        system_headers=["numpy/arrayobject.h"],
        include_dirs=[np.get_include()],
        init_code="import_array();",
        local_headers = [ "header.h", "cubature.h" ],
        arrays=[[ "n", "y"        ],
                [ "m", "vertices" ],
                [ "fdim", "result"   ]]
        )
    print "Beta Instant Compiled successfully!"
    return xpr
             
class BetaCubeAdaptive(dol.Expression):
        
    def __init__( self, cot, tol = 1e-9 ):
        
        generalInit( self, cot )

        # Generate the expression from the c++ file that
        # the input variable expresion_string refers to.
        
        # Read the code from file
        xpr_file = open( "integrate_cube.cpp" , 'r' )  
        xpr_code = xpr_file.read()
        xpr_file.close()
                
        # Compile and return an object with "integrate vector" method
        self.xpr = instant.build_module(
            code = xpr_code,
            sources = [ "helper.cpp", "hcubature.c" ],
            system_headers=["numpy/arrayobject.h"],
            include_dirs=[np.get_include()],
            init_code="import_array();",
            local_headers = [ "header.h", "cubature.h" ],
            arrays=[[ "n"   , "y"        ],
                    [ "fdim", "result"   ]]
            )
        print "Beta Cube Instant Compiled successfully!"
                
        self.tol = tol
        
    eval = generalEval
    def strictEval( self, value, y ):
        result = np.empty( 4 )
        
        self.xpr.integrateVectorOnCube( y, 
                                        self.cot.kappa, 
                                        self.tol, 
                                        result )  
            
        # The last entry is the denominator, by which we divide.
        # The first entries are the enumerators, which is a vector
        # that will later be multiplied by a unit normal
        for i in range( 3 ):
            value[i] = result[i] / result[3]

    def value_shape(self):
        return (3,)
    



#################################################################
# Radial betas that Stadler forced me to code ###################
#################################################################
        
class Beta2DRadial(dol.Expression):
    '''
    This class works the same as the Beta2D class,
    only that here we do not use the analytic expression
    for the free space fundamental solution and its
    derivative. Instead, we use the mesh approxiamtion
    obtained by the Radial class. See the module radial.py 
    for more details.
    '''
    def __init__( self, cot ):
        
        generalInit( self, cot )
                
        self.G1, self.dG1, self.G2, self.dG2 = radial.radial(cot)
    
    # As before, same business.    
    eval = generalEval

    def strictEval( self, value, y ):
        '''
        Again - this is just Riemann itegration. We integrate different
        function here though.
        '''

        ra, kappara, y_x = getRaKappara(self, y )
        
        # Recall that this is the expression for the enumerator 
        # of beta.
        ens = -( # Minus sign because does NOT cancel out here!
            self.G1( ra ) * self.dG2( ra ) +
            self.G2( ra ) * self.dG1( ra ) 
            )
        
        # Multiply to get the gradient in every vertex.
        en  = np.einsum( "i , ik -> ik", ens/ra, y_x )
        
        # Just setting the right coordinates
        self.enum0.vector().set_local( en[:,0] )
        
        # Multiply, as before, by the mass matrix to prepare for the summing ...
        self.enum0.vector().set_local( (self.M * self.enum0.vector()).array() )
        
        self.enum1.vector().set_local( en[:,1] )
        self.enum1.vector().set_local( (self.M * self.enum1.vector()).array() )
    
        # Summing, as before, amounts to Riemann integration.  Note 
        # that we DO NOT muply by kappa, since the derivatives
        # dG1 and dG2 are wrt kappa * r !!!
        en0 = np.sum( self.enum0.vector().array() )
        en1 = np.sum( self.enum1.vector().array() )

        # Same comments...
        den = self.G2( ra ) * self.G1( ra )     
        self.denom.vector().set_local( den     )
        self.denom.vector().set_local( (self.M * self.denom.vector()).array() )
        den = 2 * np.sum( self.denom.vector().array() ) # Riemann integration
        
        
        value[0] = en0 / den
        value[1] = en1 / den
     
    def value_shape(self):
        return (2,)

class Beta3DRadial(dol.Expression):
    '''
    Exactly the same idea and comments as in Beta2DRadial. 
    '''
    def __init__( self, cot ):
        
        generalInit( self, cot )
                
        self.G1, self.dG1, self.G2, self.dG2 = radial.radial(cot)
                  
    eval = generalEval
    def strictEval( self, value, y ):
        '''
        same comments as corresponding method in strictEval
        of class Beta2DRadial.
        '''
        ra, kappara, y_x = getRaKappara(self, y )
        
        ens = -( 
            self.G1( ra ) * self.dG2( ra ) +
            self.G2( ra ) * self.dG1( ra )
            )
        
        en  = np.einsum( "i , ik -> ik", ens/ra, y_x )
        
        self.enum0.vector().set_local( en[:,0] )
        self.enum0.vector().set_local( (self.M * self.enum0.vector()).array() )
        
        self.enum1.vector().set_local( en[:,1] )
        self.enum1.vector().set_local( (self.M * self.enum1.vector()).array() )
    
        self.enum2.vector().set_local( en[:,2] )
        self.enum2.vector().set_local( (self.M * self.enum2.vector()).array() )
    
        en0 = np.sum( self.enum0.vector().array() )
        en1 = np.sum( self.enum1.vector().array() )
        en2 = np.sum( self.enum2.vector().array() )

        
        den = self.G2( ra ) * self.G1( ra )    
        self.denom.vector().set_local( den     )
        self.denom.vector().set_local( (self.M * self.denom.vector()).array() )
        den = 2 * np.sum( self.denom.vector().array() )
        
        value[0] = en0 / den
        value[1] = en1 / den
        value[2] = en2 / den
     
    def value_shape(self):
        return (3,)
