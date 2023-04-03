from scipy import interpolate, spatial
from multiprocessing import Pool
from itertools import repeat
import numpy as np
import copy

__all__ = ["map_dtfe2d", "map_dtfe3d"]


def how_many(tri, num_points):
    there_are = np.where(tri.simplices == num_points)[0]
    return(there_are)


def area_triangle(points):
    return(0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) -
                        np.dot(points[:, 1], np.roll(points[:, 0], 1))))


def area_delaunay(inputs):
    tri, num = inputs
    a = area_triangle(tri.points[tri.simplices[num]])
    return(a)


def get_areas(tri, the_pool):
    shape = tri.simplices.shape[0]
    areas = np.empty(shape)
    all_inputs = zip(repeat(tri), np.arange(shape))
    areas = the_pool.map(area_delaunay, all_inputs)
    return(np.array(areas))


def vol_tetrahedron(points):
    return(abs(
        np.dot(
            (points[0] - points[3]),
            np.cross(
                (points[1] - points[3]),
                (points[2] - points[3])))) / 6.)


def vol_delaunay(inputs):
    tri, num = inputs
    a = vol_tetrahedron(tri.points[tri.simplices[num]])
    return(a)


def get_volumes(tri, the_pool):
    shape = tri.simplices.shape[0]
    volumes = np.empty(shape)
    all_inputs = zip(repeat(tri), np.arange(shape))
    volumes = the_pool.map(vol_delaunay, all_inputs)
    return(np.array(volumes))


def get_densities3d(inputs):
    tri, num, volumes = inputs
    l = how_many(tri, num)
    true_vol = np.sum(volumes[l]) / 4.
    return(1. / true_vol)


def densities3d(tri, the_pool, volumes):
    shape = tri.points.shape[0]
    dens = np.empty(shape)
    all_inputs = zip(repeat(tri), np.arange(shape), repeat(volumes))
    dens = the_pool.map(get_densities3d, all_inputs)
    return(np.array(dens))


def get_invDensities2d(inputs):
    tri, num, areas, alt = inputs
    l = how_many(tri, num)
    if alt:
        ret_val = 1. / np.sum(areas[l])
        num_tris = len(l)
        return(ret_val, num_tris)
    else:
        true_vol = np.sum(areas[l]) / 3.
    return(1. / true_vol)


def invDensities2d(tri, the_pool, areas, alt = False):
    shape = tri.points.shape[0]
    dens = np.empty(shape)
    all_inputs = zip(repeat(tri), np.arange(shape), repeat(areas), repeat(alt))
    if alt:
        full_array = np.array(the_pool.map(get_invDensities2d, all_inputs))
        return(full_array[:,0], full_array[:,1])
    else:
        dens = the_pool.map(get_invDensities2d, all_inputs)
        return(np.array(dens))


def get_densities2d(inputs):
    tri, num, areas, alt = inputs
    l = how_many(tri, num)
    if alt:
        ret_val = np.sum(areas[l])
        num_tris = len(l)
        return(ret_val, num_tris)
    else:
        true_vol = np.sum(areas[l]) / 3.
    return(true_vol)


def densities2d(tri, the_pool, areas, alt = False):
    shape = tri.points.shape[0]
    dens = np.empty(shape)
    all_inputs = zip(repeat(tri), np.arange(shape), repeat(areas), repeat(alt))
    if alt:
        full_array = np.array(the_pool.map(get_densities2d, all_inputs))
        return(full_array[:,0], full_array[:,1])
    else:
        print('else')
        dens = the_pool.map(get_densities2d, all_inputs)
        return(np.array(dens))


def map_dtfe3d(x, y, z, xsize, ysize=None, zsize=None):
    """
    Create a 3d density cube from given x, y and z points in a volume

    Parameters
    ----------
    x : An N-length one dimensional array
    The x coordinates of the point distribution
    y : An N-length one dimensional array
    The y coordinates of the point distribution
    z : An N-length one dimensional array
    The z coordinates of the point distribution
    xsize : Integer
    The x dimension of the cube
    ysize : Integer, optionale
    The y dimension of the cube. If ysize is not given, it assumes that the x, y and z axis share the same dimension
    zsize : Integer, optional
    The z dimension of the cube. If zsize is not given, it assumes that the x, y and z axis share the same dimension

    Returns
    -------
    grid : An (xsize, ysize, zsize)-shaped array
    The density cube in 3d

    """
    tab = np.vstack((x, y, z)).T
    tri = spatial.Delaunay(tab)
    the_pool = Pool()
    volumes = get_volumes(tri, the_pool)
    d = densities3d(tri, the_pool, volumes)
    the_pool.close()
    if (ysize is None) & (zsize is None):
        size = xsize
        x_m = np.linspace(np.min(x), np.max(x), size)
        y_m = np.linspace(np.min(y), np.max(y), size)
        z_m = np.linspace(np.min(z), np.max(z), size)
    else:
        x_m = np.linspace(np.min(x), np.max(x), xsize)
        y_m = np.linspace(np.min(y), np.max(y), ysize)
        z_m = np.linspace(np.min(z), np.max(z), zsize)
    x_m, y_m, z_m = np.meshgrid(x_m, y_m, z_m)
    grid = interpolate.griddata(tab, d, (x_m, y_m, z_m), method='linear')
    return(grid)


def map_dtfe2d(x, y, xsize=None, ysize=None, inverse=False):
    """
    Create a 2d density map from given x and y points in a plan

    Parameters
    ----------
    x : An N-length one dimensional array
    The x coordinates of the point distribution
    y : An N-length one dimensional array
    The y coordinates of the point distribution
    xsize : Integer, optional
    The x dimension of the map. If xsize if not given, the algorithm compute a best resolution in x and y,
    taking statistically a pixel size under ten times the square root of the 5 sigma area. The algorithm
    try to give the same resolution in x and y
    ysize : Integer, optional
    The y dimension of the map. If ysize is not given, ysize take the value of xsize, if given

    Returns
    -------

    grid : An 2 dimensional array
    The density map in 2d

    """
    tab = np.vstack((x, y)).T
    tri = spatial.Delaunay(tab)
    the_pool = Pool()
    areas = get_areas(tri, the_pool)
    if inverse:
        d = invDensities2d(tri, the_pool, areas)
    else:
        d = densities2d(tri, the_pool, areas)
    the_pool.close()
    if (xsize is None) & (ysize is None):
        if y.max()-y.min() < x.max()-x.min():
            xsize = int(1/(np.sqrt(1/(np.median(d)+5*np.std(d)))/(10/((y.max()-y.min())/(x.max()-x.min())))))
            ysize = int(1/(np.sqrt(1/(np.median(d)+5*np.std(d)))/10))
        else:
            xsize = int(1/(np.sqrt(1/(np.median(d)+5*np.std(d)))/10))
            ysize = int(1/(np.sqrt(1/(np.median(d)+5*np.std(d)))/(10*((y.max()-y.min())/(x.max()-x.min())))))
        x_m = np.linspace(np.min(x), np.max(x), xsize)
        y_m = np.linspace(np.min(y), np.max(y), ysize)
    elif (xsize is not None) & (ysize is None):
        size = xsize
        x_m = np.linspace(np.min(x), np.max(x), size)
        y_m = np.linspace(np.min(y), np.max(y), size)
    else:
        x_m = np.linspace(np.min(x), np.max(x), xsize)
        y_m = np.linspace(np.min(y), np.max(y), ysize)
    x_m, y_m = np.meshgrid(x_m, y_m)
    grid = interpolate.griddata(tab, d, (x_m, y_m), method='linear')
    return(grid)

def checkEdges(tri, graph, complete = True):
    tri_ex = copy.deepcopy(tri)
    nodeVal = list(graph.nodes)
    include_list = []
    if complete:
        for simplex in tri_ex.simplices:
            include_simplex = graph.has_edge(nodeVal[simplex[0]], nodeVal[simplex[1]])
            include_simplex *= graph.has_edge(nodeVal[simplex[1]], nodeVal[simplex[2]])
            include_simplex *= graph.has_edge(nodeVal[simplex[2]], nodeVal[simplex[0]])
            include_list += [include_simplex]
    else:
        for simplex in tri_ex.simplices:
            include_simplex = graph.has_edge(nodeVal[simplex[0]], nodeVal[simplex[1]])
            include_simplex += graph.has_edge(nodeVal[simplex[1]], nodeVal[simplex[2]])
            include_simplex += graph.has_edge(nodeVal[simplex[2]], nodeVal[simplex[0]])
            include_list += [min(include_simplex, 1)]        
            
    tri_ex.simplices = tri_ex.simplices[np.array(include_list).astype(bool)]
    return(tri_ex)
        
    
def spec_dtfe2d(x, y, graph = None, inverse=False, complete = True):
    """
    Calculates the DTFE density at each x,y coord.

    Parameters
    ----------
    x : An N-length one dimensional array
    The x coordinates of the point distribution
    y : An N-length one dimensional array
    The y coordinates of the point distribution
    graph : A networkx graph
    This graph defines all possible edges to use
    in the Delaunay Tesselation
    
    Returns
    -------

    The total area for each point and the number of triangles which comprise
    that area.

    """
    tab = np.vstack((x, y)).T
    tri = spatial.Delaunay(tab)
    if graph is not None:
        tri = checkEdges(tri, graph, complete = complete)
    
    the_pool = Pool()
    areas = get_areas(tri, the_pool)
    if inverse:
        tot_areas, num_tris = invDensities2d(tri, the_pool, areas, alt = True)
    else:
        tot_areas, num_tris = densities2d(tri, the_pool, areas, alt = True)
    the_pool.close()
    return(tot_areas, num_tris)

def get_triangle_vals(tri):
    """
    Calculates the DTFE density at each x,y coord.

    Parameters
    ----------
    tri : A Delaunay object
    This is a tesselation object for which the triangle areas will be 
    calculated.
    
    Returns
    -------

    list of triangle areas

    """
    the_pool = Pool()
    areas = get_areas(tri, the_pool)
    the_pool.close()
    return(areas)



