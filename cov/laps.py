from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt

def project( v, subspace ):
    return v[0, subspace-1]

def pad(v, space):
    n = length(v)
    z = np.zeros( space - n )
    return np.concatenate( (v,z) , axis= 1 )

def nufft( v, domainSize, junk size, idCoef, power ):
    


 
# do everything in [-1,1]^2
perimeter = np.array([ [ -0.2 , 0.45 ] , 
                       [  0.33, 0.34 ] ,
                       [  0.1 ,-0.67 ] ,
                       [ -0.67,-0.66 ] ])
                       
interior =  np.array([ [ 0    , 0    ] ,
                       [-0.21 ,-0.25 ] ,
                       [-0.2  , 0.25 ] ,
                       [-0.35 ,-0.5  ] ,
                       [ 0.02 ,-0.31 ] ,
                       [-0.2  , 0.0  ] ,
                       [ 0.1  , 0.1  ] ]) 

exterior =  np.array([ [  1   , 1    ] ,
                       [ -1   , 1    ] ,
                       [  1   ,-1    ] ,
                       [ -1   ,-1    ] ,
                       [ 0.56 , 0.7  ] , # top right
                       [ 0.3  ,-0.6  ] , # 
                       [-0.6  , 0.4  ] , # 
                       [-0.8  ,-0.5  ] ])# bottom left


points = np.concatenate( (perimeter, interior, exterior) , axis=0 )


#tri = Delaunay( perimeter , incremental = True )
#tri.add_points(interior)
#tri.add_points(exterior)
if True:
    plt.plot(interior[:, 0], interior[ :,1], 'd' , color = 'g')
    plt.plot(exterior[:, 0], exterior[ :,1], 'd' , color = 'b')
    plt.plot(perimeter[:,0], perimeter[:,1], 'o', color= 'r')
    plt.title(" red = bdry, green = interior, blue = extrerior " )
    plt.show()
    
