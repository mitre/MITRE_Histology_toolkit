from matplotlib import collections
from descartes import PolygonPatch
from shapely import geometry, ops
from scipy import spatial
import numpy as np
import pylab as pl

def plot_polygon(polygon):
    fig = pl.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    margin = 50
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return fig

def alpha_shape(points, alpha):
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
           # already added
           return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
    
    coords = np.array([point.coords[0]
                       for point in points])
    tri = spatial.Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = np.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = np.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = np.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(ops.polygonize(m))
    return(ops.unary_union(triangles), edge_points)

def plotComparison(coords, alpha = 0.003):
    """
    Plots the convex vs the concave hulls
    Parameters
    ----------
    coords : pandas DataFrame
        A dataframe with columns x and y of the coordinates.
    alpha : float
        The the sensitivity parameter for the contours.
        The inverse of the maximum perimeter.
    """
    points=[]
    for line in range(coords.shape[0]):
        points.append(geometry.shape(geometry.Point(coords.x[line],coords.y[line])))
    
    x = [p.x for p in points]
    y = [p.y for p in points]
    
    pl.figure(figsize=(8,8))
    point_collection = geometry.MultiPoint(list(points))
    point_collection.envelope
    convex_hull_polygon = point_collection.convex_hull
    _ = plot_polygon(convex_hull_polygon)
    _ = pl.plot(x,y,'o', color='#f16824')
    concave_hull, edge_points = alpha_shape(points, alpha=alpha)
    lines = collections.LineCollection(edge_points)  # noqa
    _ = plot_polygon(concave_hull)       
    _ = pl.plot(x,y,'o', color='#f16824')

def get_concave_area(coords, alpha = 0.003):
    """
    Returns the area of the concave hull for the coordinates supplied.

    Parameters
    ----------
    coords : pandas DataFrame
        A dataframe with columns x and y of the coordinates.
    alpha : float
        The the sensitivity parameter for the contours.
        The inverse of the maximum perimeter.

    Returns
    -------
    The area in pixels squared.
    """
    points=[]
    for line in range(coords.shape[0]):
        points.append(geometry.shape(geometry.Point(coords.x[line],coords.y[line])))

    concave_hull, edge_points = alpha_shape(points, alpha=alpha)
    
    if concave_hull.area == 0:
        print("Warning: alpha set too high, concave hull area is 0")
    
    return(concave_hull.area)

