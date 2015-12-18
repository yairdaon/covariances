from dolfin import *
import numpy as np
import math
import pdb

def get_normals( mesh ):
    
    normals = np.ones( mesh.shape )

    for i in range( bdry.shape[0] ):
        x = bdry[i,:]
        
        if abs( x[0] ) < 1E-5:
            normals[i,1] = -1.0 
        elif abs( x[0] - 1 ) < 1E-5:
            normals[i,1] = 1.0
        else: 
            normals[i,1] = 0.0

        if abs( x[1] ) < 1E-5:
            normals[i,0] = -1.0 
        elif abs( x[1] - 1 ) < 1E-5:
            normals[i,0] = 1.0
        else: 
            normals[i,0] = 0.0
        
        norms        = np.linalg.norm( normals, axis=1 )
        normals[:,0] = normals[:,0] / norms
        normals[:,1] = normals[:,1] / norms

        assert np.allclose( np.linalg.norm( normals, axis=1 ), 1.0 )
        
        return normals


def get_poisson_betas( bdry, bdry_ind, normals, interior):

    betas = np.zeros(  )
    normals = get_normals(mesh)

    for i in bdry_ind:

        x
        y_minus_x           = interior.T - x
        y_minus_x_sqr       = y_minus_x * y_minus_x
        y_minus_x_norms_sqr = np.sum( y_minus_x_sqr, axis = 1)
        y_minus_x_norms     = np.sqrt( y_minus_x_norms_sqr )
        y_minus_x_dot_n     = np.einsum( "ij , j -> i", y_minus_x, normal )
        

        K0 = -np.log( y_minus_x_norms )
        K0_sqr = K0 * K0
        denominator = np.sum( K0_sqr )
        
        dm_dn =   y_minus_x_dot_n / ( y_minus_x_norms * y_minus_x_norms )
        
        enumerator = np.sum( K0 * dm_dn  )
    
        beta = -enumerator / denominator
    
    return betas

def ones_bdry_zeros_int( mesh ):
    '''
    array of the size of the number
    of nodes in  ...

    Copied from
    http://fenicsproject.org/qa/2989/vertex-on-mesh-boundary
    '''

    # Mark a CG1 Function with ones on the boundary
    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, 1, DomainBoundary())
    u = Function(V)
    bc.apply(u.vector())

    # Get vertices sitting on boundary
    d2v = dof_to_vertex_map(V)
    vertices_on_boundary = d2v[u.vector() == 1.0]

    # Mark VertexFunction to check
    vf = VertexFunction('size_t', mesh, 0)
    vf.array()[vertices_on_boundary] = 1

    return vf

def get_bdry_indices( mesh ):
    '''
    return an array of indices that tells you which
    nodes are on the boundary
    '''

    # Mark a CG1 Function with ones on the boundary
    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, 1, DomainBoundary())
    u = Function(V)
    bc.apply(u.vector())

    # Get vertices sitting on boundary
    d2v = dof_to_vertex_map(V)
    vertices_on_boundary = d2v[u.vector() == 1.0]

    return vertices_on_boundary

def get_int_indices( mesh ):
    '''
    returns indices of the interior
    '''
     # Mark a CG1 Function with ones on the boundary
    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, 1, DomainBoundary())
    u = Function(V)
    bc.apply(u.vector())

    # Get vertices sitting on boundary
    d2v = dof_to_vertex_map(V)
    vertices_in_interior = d2v[u.vector() == 0.0]

    return vertices_in_interior
    
def get_interior_coordinates( mesh ):

    indices = int_indices( mesh )
    coo = mesh.coordinates()
    
    return coo[ indices,: ]

def get_bdry_coordiantes( mesh ):

    indices = bdry_indices( mesh )
    coo = mesh.coordinates()
    
    return coo[ indices,: ]



if __name__ =="__main__":
   
    m = n = 5
    lam = 1.23

    mesh = UnitSquareMesh( m-1, n-1 )

    V = FunctionSpace ( mesh, "CG", 1 )
    u = TrialFunction ( V )
    v = TestFunction ( V )
    beta = Function( V )
    
    data = np.empty( (m*n,) )
    data[get_bdry_indices( mesh )] = 2.
    data[get_int_indices( mesh )] = -1.
    
    beta.vector().set_local( data ) 
    
    boundary_parts =  MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    class RobinBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-5
            logical = abs( x[0] ) < tol
            logical = logical or abs(x[0]-1) < tol
            logical = logical or abs(x[1]-1) < tol
            logical = logical or abs(x[1])   < tol
            
            return on_boundary and logical

    Gamma_R = RobinBoundary()
    Gamma_R.mark(boundary_parts, 0)
      
    #  Define the bilinear form, the left hand side of the FEM system.
    AUV = inner( grad ( u ), grad ( v ) ) * dx + u*v*ds(0)

    #pdb.set_trace()
    #  Define the linear form, the right hand side. Later we add point sources.
    LV = Constant ( 0.0 ) * v * dx

    #  Assemble the system matrix and right hand side vector, 
    A = assemble ( AUV ) #, exterior_facet_domains=boundary_parts)
    b = assemble( LV )

    #  Modify the right hand side vector to account for point source(s)
    delta = PointSource ( V, Point ( 0.01, 0.5 ), 10.0 )
    delta.apply ( b )
    gamma = PointSource ( V, Point( 0.95, 0.05 ), 10.0 )
    gamma.apply( b )
    eta = PointSource ( V, Point( 0.5, 0.5 ), 10.0 )
    eta.apply( b )

    #  Solve the linear system for u.
    u = Function ( V )              
    solve ( A, u.vector(), b )

    # file = File( "vis/poisson" + filename + ".pvd")
    # file << u
    
    # Plot the solution.
    plot ( u, interactive = True ) 



