import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math
import os

import parameters

pts ={}
pts["square"]           = [ np.array( [ 0.05  , 0.5   ] ) ]
pts["parallelogram"]    = [ np.array( [ 0.025 , 0.025 ] ) ]
pts["dolfin_coarse"]    = [ np.array( [ 0.45  , 0.65  ] ) ] 
pts["dolfin_fine"]      = [ np.array( [ 0.45  , 0.65  ] ) ] 
pts["pinch"]            = [ np.array( [ 0.35  , 0.155 ] ) ]
pts["l_shape"]          = [ np.array( [ 0.45  , 0.65  ] ),
                            np.array( [ 0.995 , 0.2   ] ),
                            np.array( [ 0.05  , 0.005 ] ) ]

no_scaling =  lambda x: 1.0
def apply_sources ( container, b, scaling = no_scaling ):
    sources = pts[container.mesh_name]
    for source in sources:
        PointSource(
            container.V, 
            Point ( source ),
            scaling(source)
        ).apply( b )

def refine( mesh_name, show = False ):
    
    mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )

    # Break point
    p   = Point( helper.pts[mesh_name][0] )
    tol = 0.05

    # Selecting edges to refine
    class Border(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], p.x(), tol) and near(x[1], p.y(), tol)
        
    Border = Border()

    # Number of refinements
    nor = 2

    # refine!!!
    for i in range(nor):
        edge_markers = EdgeFunction("bool", mesh_obj)
        Border.mark(edge_markers, True)
        adapt(mesh_obj, edge_markers)
        mesh_obj = mesh_obj.child()
        
    if show:
        plot(mesh_obj, interactive=True)
    
    return mesh_obj

def set_vg( container, BC ):

    u = container.u
    v = container.v
    kappa2 = container.kappa2
    kappa = container.kappa
    normal = container.normal

    if "mixed_robin" in BC:
        mix_beta = parameters.Robin( container, "mix_enum", "mix_denom" )
        a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx + inner( mix_beta, normal )*u*v*ds
        A = assemble( a )
        
    elif "improper_robin" in BC:
        imp_beta = parameters.Robin( container, "imp_enum", "imp_denom" )
        a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx + inner( imp_beta, normal )*u*v*ds
        A = assemble(a)
        
    elif "naive_robin" in BC:
        a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx + 1.42*kappa*u*v*ds
        A = assemble( a )
    
    elif "neumann" in BC:
        a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx
        A = assemble( a )

    elif "dirichlet" in BC:
        def boundary(x, on_boundary):
            return on_boundary
        f = Constant( 0.0 )
        bc = DirichletBC(container.V, f, boundary)
        a = inner(grad(u), grad(v))*dx + kappa2*u*v*dx 
        A, _ = assemble_system ( a, f*v*dx, bc )
    else:
        raise ValueError( "Boundary condition type not supported. Go home." )
        
    container._variances[BC], container._gs[BC] = get_var_and_g( container, A )


def get_var_and_g( container, A ):
    
    var   = Function( container.V )
    g     = Function( container.V )
    n     = container.n
    
    if container.num_samples > 0:
        tmp   = Function( container.V )
        noise = Function( container.V )
        
        for i in range( container.num_samples ):
        
            # White noise, weighted by mass matrix.
            noise.vector().set_local( np.einsum( "ij, j -> i", container.sqrt_M, np.random.normal( size = n ) ) )
            
            # Apply inverse operator to white noise.
            solve( A, tmp.vector(), noise.vector() )

            # Variance is averaged sum of squares.
            var.vector().set_local( var.vector().array() + tmp.vector().array()*tmp.vector().array() )

        # Divid by number of samples to get average sum of squares.
        var.vector().set_local( var.vector().array() / container.num_samples )
    
    else:
        
        V = container.V
        mesh_obj = container.mesh_obj
        tmp1 = Function( V )
        tmp2 = Function( V )

        b    = assemble( Constant(0.0) * container.v * dx )
                     
        coor = V.dofmap().tabulate_all_coordinates(mesh_obj)
        coor.resize(( V.dim(), container.dim ))

        vertex_values = np.zeros(mesh_obj.num_vertices())
            
        for vertex in vertices( mesh_obj ):
              
            pt = Point( vertex.x(0), vertex.x(1) )
         
            # Apply point source
            PointSource( V, pt, 1.0 ).apply( b )

            # Apply inverse laplacian once ...
            solve( A, tmp1.vector(), b )

            # ... and twice (and reassemble!!!)
            solve( A, tmp2.vector(), assemble( tmp1 * container.v * dx ) )
                    
            # Place the value where it belongs.
            vertex_values[vertex.index()] = tmp2.vector().array()[vertex_to_dof_map(V)[vertex.index()]]
            
            # Remove point source
            PointSource( V, pt, -1.0 ).apply( b )
                    
        var.vector()[:] = vertex_values[dof_to_vertex_map(V)]

        
    g.vector().set_local( 
        np.sqrt( container.sig2 / var.vector().array() )
    )

    return var, g 
    
    
def save_plots( data, 
                desc,
                mesh_name,
                mode = "color",
                ran = [None, None],
                scalarbar = False ):

    source = pts[mesh_name][0]
    x_range = np.arange( 0.0, 0.5, 0.005 )
    y = [] 
    x = []

    plot_file = "../../PriorCov/" + mesh_name + desc[0].replace(" ","") + desc[1].split(" ")[0] + ".txt"
    try:
        os.remove( plot_file )
    except:
        pass
        
    if "square" in mesh_name:
        if "Greens" in desc[1]:
            slope = 0.0
            intercept = source[1] - slope * source[0]
            line = lambda x: (x, slope * x + intercept )

            for pt in x_range:
                try:
                    add_point( plot_file, pt, data( line(pt) ) )
                except:
                    pass # so the point isn't in the domain. So what? Just skip!
                   
            
    elif "parallelogram" in mesh_name:
        if "Greens" in desc[1]:
            slope = .9
            intercept = source[1] - slope * source[0]
            line = lambda x: (x, slope * x + intercept )
               
            for pt in x_range:
                try:
                    add_point( plot_file, pt, data( line(pt) ) )
                except:
                    pass # The poin't isnt in the domain. So what? Just skip!
                
                    
    elif "dolfin" in mesh_name:
        
        file_name =  mesh_name + "_" + desc[0].replace( " ", "_" ) + desc[1].replace( " ", "_" )
        
        if "Fundamental" in desc[1]:
            plotter = plot( data, 
                            title = desc[0],
                            mode = mode,
                            range_min = ran[0],
                            range_max = ran[1],
                            interactive = False,
                            scalarbar = True,
                            window_height = 500,
                            window_width = 600
                            )

        else:
            plotter = plot( data, 
                            title = desc[0] + " " + desc[1],
                            mode = mode,
                            range_min = ran[0],
                            range_max = ran[1],
                            interactive = False,
                            scalarbar = scalarbar,
                            window_height = 500,
                            window_width = 500
                            )
        plotter.zoom( 1.4 )
        plotter.write_png( "../../PriorCov/" + file_name )
                    
    else:
        plot( data, title = desc[0] + "_" + desc[1] )

       
def update_x_xp( x, xp ):
    xp.x[0] = x[0]
    xp.x[1] = x[1]

def make_2D_parallelogram( m, n, s = 1.6 ):
        
    par = UnitSquareMesh( m,n )

    # First make a denser mesh towards r=a
    x = par.coordinates()[:,0]
    y = par.coordinates()[:,1]

    x = np.power( x, s )
    y = np.power( y, s )
    
    A = np.array( [ [ 2 , 1 ],
                    [ 1 , 2 ] ] )
    
    xy = np.array([x, y])
    par.coordinates()[:] = np.einsum( "ij, jk -> ki", A, xy )

    return par


def add_point( plot_file, x, y ):
    dat = str(x) + "   " + str(y) + "\n"
    with open( plot_file, "a") as myfile:
        myfile.write(dat)
                               
    
