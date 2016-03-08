import numpy as np
from scipy import special as sp
from dolfin import *
import pdb
import math
from matplotlib import pyplot as plt
import os

pts ={}
pts["square"]           = [ np.array( [ 0.05  , 0.5   ] ) ]
pts["dolfin_coarse"]    = [ np.array( [ 0.45  , 0.65  ] ) ] 
pts["dolfin_fine"]      = [ np.array( [ 0.45  , 0.65  ] ) ] 
pts["pinch"]            = [ np.array( [ 0.35  , 0.155 ] ) ]
pts["l_shape"]          = [ np.array( [ 0.45  , 0.65  ] ),
                            np.array( [ 0.995 , 0.2   ] ),
                            np.array( [ 0.05  , 0.005 ] ) ]

color_counter = 0
colors = [ 'g' , 'b' , 'r', 'k', 'c' , 'm', 'y' ] 


no_scaling =  lambda x: 1.0
def apply_sources ( container, b, scaling = no_scaling ):
    sources = pts[container.mesh_name]
    for source in sources:
        PointSource( container.V, Point ( source ), scaling(source)  ).apply( b )


def get_var_and_g( container, A ):
    
    n     = container.n
    tmp   = Function( container.V )
    noise = Function( container.V )
    var   = Function( container.V )
    g     = Function( container.V )
    
    for i in range( container.num_samples ):
        
        noise.vector().set_local( np.einsum( "ij, j -> i", container.sqrt_M, np.random.normal( size = n ) ) )
        
        solve( A, tmp.vector(), noise.vector() )
        var.vector().set_local( var.vector().array() + tmp.vector().array()*tmp.vector().array() )
                
    var.vector().set_local( var.vector().array() / container.num_samples )
            

    g.vector().set_local( 
        np.sqrt( container.sig2 / var.vector().array() )
    )

    return var, g 
 
def save_plots( data, 
                title,
                mesh_name,
                mode = "color",
                ran = [],
                scalarbar = False ):
    
    if mesh_name == "square":
        if "Greens Function" in title or "Fundamental" in title:
        
            global color_counter
            x = np.arange(0.0, 0.5, 0.01 )
            y = []
            
            for pt in x:
                y.append( data( (pt,0.5) ) ) 
        
            plt.plot( x,y, colors[color_counter], label = title )
            color_counter = color_counter + 1
        
    elif "dolfin" in mesh_name:
        
        file_name =  mesh_name + "_" + title.replace( " ", "_" )

        if ran == []:
            plot( data, 
                  title = title,
                  mode = mode,
                  interactive = False,
                  scalarbar = scalarbar,
              ).write_png( "../../PriorCov/" + file_name )

        elif len(ran) == 2:
            plot( data, 
                  title = title,
                  mode = mode,
                  range_min = ran[0],
                  range_max = ran[1],
                  interactive = False,
                  scalarbar = scalarbar,
              ).write_png( "../../PriorCov/" + file_name )
       
        else:
            raise NameError( "Range is not empty, neither it has two entries" )
            
    else:
        plot( data, title = title )

    #print "Maximum of " + title + " = " + str( np.amax( data.vector().array() ) )


def save_plots( data, title, mesh_name, mode = "color", ran = [] ):

    file_name =  mesh_name + "_" + title.replace( " ", "_" )
    
    if ran == []:
        plot( data, 
              title = title,
              mode = mode,
          ).write_png( "../../PriorCov/" + file_name )

    elif len(ran) == 2:
        plot( data, 
              title = title,
              mode = mode,
              range_min = ran[0],
              range_max = ran[1],
          ).write_png( "../../PriorCov/" + file_name )
       
    else:
        raise NameError( "Range is not empty, neither it has two entries" )
   
    File( "data/" + file_name + ".pvd") << data


def plot_variable(u, name, direc, cmap='gist_yarg', scale='lin', numLvls=12,
                  umin=None, umax=None, tp=False, tpAlpha=0.5, show=True,
                  hide_ax_tick_labels=False, label_axes=True, title='',
                  use_colorbar=True, hide_axis=False, colorbar_loc='right'):
    """
    """
    mesh = u.function_space().mesh()
    v    = u.compute_vertex_values(mesh)
    x    = mesh.coordinates()[:,0]
    y    = mesh.coordinates()[:,1]
    t    = mesh.cells()

    d    = os.path.dirname(direc)
    if not os.path.exists(d):
        os.makedirs(d)

    if umin != None:
        vmin = umin
    else:
        vmin = v.min()
    if umax != None:
        vmax = umax
    else:
        vmax = v.max()

    # countour levels :
    if scale == 'log':
        v[v < vmin] = vmin + 1e-12
        v[v > vmax] = vmax - 1e-12
        from matplotlib.ticker import LogFormatter
        levels      = np.logspace(np.log10(vmin), np.log10(vmax), numLvls)
        formatter   = LogFormatter(10, labelOnlyBase=False)
        norm        = colors.LogNorm()

    elif scale == 'lin':
        v[v < vmin] = vmin + 1e-12
        v[v > vmax] = vmax - 1e-12
        from matplotlib.ticker import ScalarFormatter
        levels    = np.linspace(vmin, vmax, numLvls)
        formatter = ScalarFormatter()
        norm      = None
        
    elif scale == 'bool':
        from matplotlib.ticker import ScalarFormatter
        levels    = [0, 1, 2]
        formatter = ScalarFormatter()
        norm      = None

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111)

    c = ax.tricontourf(x, y, t, v, levels=levels, norm=norm,
                       cmap=pl.get_cmap(cmap))
    plt.axis('equal')

    if tp == True:
        p = ax.triplot(x, y, t, '-', lw=0.2, alpha=tpAlpha)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    if label_axes:
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
    if hide_ax_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if hide_axis:
        plt.axis('off')

    # include colorbar :
    if scale != 'bool' and use_colorbar:
        divider = make_axes_locatable(plt.gca())
        cax  = divider.append_axes(colorbar_loc, "5%", pad="3%")
        cbar = plt.colorbar(c, cax=cax, format=formatter,
                            ticks=levels)
        tit = plt.title(title)

    if use_colorbar:
        plt.tight_layout(rect=[.03,.03,0.97,0.97])
    else:
        plt.tight_layout()
    plt.savefig(direc + name + '.png', dpi=300)
    if show:
        plt.show()
    plt.close(fig)
