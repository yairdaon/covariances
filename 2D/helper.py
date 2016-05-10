import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math
import os

pts ={}
pts["square"]           = np.array( [ 0.05    , 0.5   ] ) 
pts["tmp"]              = np.array( [ 0.05    , 0.5   ] ) 
pts["parallelogram"]    = np.array( [ 0.025   , 0.025 ] ) 
pts["dolfin"]           = np.array( [ 0.45    , 0.65  ] ) 
pts["pinch"]            = np.array( [ 0.35    , 0.155 ] ) 
pts["antarctica"]       = np.array( [ -1850.0 , 900.0 ] )
pts["l_shape"]          = np.array( [ 0.45    , 0.65  ] )
pts["antarctica"]       = np.array( [ 0.0     , 0.0   ] )


no_scaling =  lambda x: 1.0
def apply_sources ( container, b, scaling = no_scaling ):
   
    source = get_source( container.mesh_name )
    PointSource( container.V, 
                 Point ( source ),
                 scaling( source )
             ).apply( b )

def get_source( mesh_name ):
    if "antarctica" in mesh_name:
        return pts["antarctica"]
    elif "dolfin" in mesh_name:
        return pts["dolfin"]
    else:
        return pts[mesh_name]
        
def refine( mesh_name, nor = 2, show = False ):
    
    if "square" in mesh_name:
        mesh_obj = UnitSquareMesh( 50, 50 )
        tol = 0.1
    elif "parallelogram" in mesh_name:
        mesh_obj = make_2D_parallelogram( 50, 50, 1.4 )
        tol = 0.35
    elif "antarctica" in mesh_name:
        mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )
        tol = 1e2
    else:
        mesh_obj = Mesh( "meshes/" + mesh_name + ".xml" )
        tol = 0.05
    
    p  = Point( get_source( mesh_name ) )
 
    # Selecting edges to refine
    class Border(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], p.x(), tol) and near(x[1], p.y(), tol)
        
    Border = Border()

    # refine!!!
    for i in range(nor):
        edge_markers = EdgeFunction("bool", mesh_obj)
        Border.mark(edge_markers, True)
        adapt(mesh_obj, edge_markers)
        mesh_obj = mesh_obj.child()
        tol = 0.5 * tol
        
    if show:
        plot(mesh_obj, interactive=True)
    
    return mesh_obj



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
        
        solver = LUSolver(A)
        solver.parameters['reuse_factorization'] = True 
           
        V = container.V
        mesh_obj = container.mesh_obj
        tmp1 = Function( V )
        tmp2 = Function( V )

        b    = assemble( Constant(0.0) * container.v * dx )
      
        vertex_values = np.zeros(mesh_obj.num_vertices())
          
        total = 0

        # Iterate over vertices according to their indices
        for vertex in vertices( mesh_obj ):
      
            pt = Point( vertex.x(0), vertex.x(1) )
         
            # Apply point source
            PointSource( V, pt, 1.0 ).apply( b )

            # Apply inverse laplacian once ...
            solver.solve( tmp1.vector(), b )

            # ... and twice (and reassemble!!!)
            solver.solve( tmp2.vector(), assemble( tmp1 * container.v * dx ) )
               
            # Place the value where it belongs.
            vertex_values[vertex.index()] = tmp2.vector().array()[vertex_to_dof_map(V)[vertex.index()]]
            
            # Remove point source
            PointSource( V, pt, -1.0 ).apply( b )
             
        var.vector().set_local( vertex_values[dof_to_vertex_map(V)] )

        
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

    source = get_source( mesh_name )
    x_range = np.arange( 0.0, 0.5, 0.01 )
    y = [] 
    x = []

    plot_file   = "../../PriorCov/" + mesh_name + "_" + desc[0].replace(" ","_") + "_" + desc[1].replace(" ", "_") + ".txt"
    line_file   = "../../PriorCov/" + mesh_name + "_Line.txt"
    source_file = "../../PriorCov/" + mesh_name + "_Source.txt" 
    try:
        os.remove( plot_file )
        os.remove( line_file )
        os.remove( source_file )
    except:
        pass
        
    if "square" in mesh_name or "parallelogram" in mesh_name:
        if "Greens" in desc[1]:

            add_point( source_file, source[0], source[1] )
            if "square" in mesh_name:
                slope = 0.0
            else:
                slope = .6

            intercept = source[1] - slope * source[0]
            line = lambda x: (x, slope * x + intercept )

            for pt in x_range:
                try:
                    add_point( plot_file, pt, data( line(pt) ) )
                    add_point( line_file, pt, line(pt)[1] )
                except:
                    pass
  
    elif "dolfin" in mesh_name:
        
        file_name =  mesh_name + "_" + desc[0].replace( " ", "_" ) + "_" + desc[1].replace( " ", "_" )
        
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

    elif "antarctica" in mesh_name:
        
        file_name =  mesh_name + "_" + desc[0].replace( " ", "_" ) + "_" + desc[1].replace( " ", "_" )
        
        plotter = plot( data, 
                        title = desc[0],
                        mode = mode,
                        interactive = False,
                        scalarbar = True,
                        window_height = 500,
                        window_width = 600 )
        plotter.write_png( "../../PriorCov/" + file_name )

    else:
        plot( data, title = desc[0] + "_" + desc[1], interactive = True )

       
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
    
    A = np.array( [ [ 2.5 , 1   ],
                    [ 1   , 2.5 ] ] )
    
    xy = np.array([x, y])
    par.coordinates()[:] = np.einsum( "ij, jk -> ki", A, xy )

    file_name = "../../PriorCov/parallelogram.txt"
    try:
        os.remove( file_name )
    except:
        pass

    pts = [ np.array( [ 0.0, 0.0 ] ),
            np.array( [ 0.0, 1.0 ] ),
            np.array( [ 1.0, 1.0 ] ),
            np.array( [ 1.0, 0.0 ] ),
            np.array( [ 0.0, 0.0 ] )]
    for pt in pts:
        new_pt = np.dot( A, pt )
        add_point( file_name, new_pt[0], new_pt[1] )
            
    return par


def add_point( plot_file, x, y ):
    dat = str(x) + "   " + str(y) + "\n"
    with open( plot_file, "a") as myfile:
        myfile.write(dat)
                               
    
