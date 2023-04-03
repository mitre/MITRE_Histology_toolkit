#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The DLTrace class is designed to find the traces of the most
densely populated areas of a space. The algorithm uses Delaunday 
tessellation and the user can choose to either use the longest edge
of the simplices or the Delaunay Tessellation Field Estimator (DTFE)
to determine which points to retain.

##EXAMPLE:
    import DLTrace
    
    #Create a set of points to work with
    x=np.random.randint(-1000, 1000, 500)
    y=np.random.randint(-1000, 1000, 500)

    for m in range(-5, 5):
        x1=np.arange(-50, 50) + np.random.randn(100)*2.
        y1=np.random.randint(180)+m*x1 + np.random.randn(100)*3
        x=np.concatenate((x,x1))
        y=np.concatenate((y, y1))
        x1=np.arange(100, 200) + np.random.randn(100)*2.
        y1=np.random.randint(180)+m*x1 + np.random.randn(100)*3
        x=np.concatenate((x,x1))
        y=np.concatenate((y, y1))
        
    myDL=DLTrace(x,y, buffer=1.0)
    myDL.run()
    ax = myDL.traces.plot()
    ax.plot(x,y, '.')
    
Created on Mon Apr 11 09:48:38 2022

@author: jrthompson
"""

#====================== LIBRARIES ==================================#

from shapely.geometry import MultiLineString
import numpy as np
import pandas as pd
from scipy import spatial
import geopandas as gp
from shapely.ops import unary_union, polygonize
from sklearn.mixture import GaussianMixture


#====================== CLASS DEFINITION ==========================#

class DLTrace(object):
    """The DLTrace class provides functions for creating Delaunay
    tessellations from a set of points. By converting a set of the
    Delaunay simplices to polygons the class facilitates segmenting
    the space into areas of higher density points (i.e., traces). The
    intended use to find things like shipping lanes or areas of
    high air traffic etc.
    
    Parameters:
    ----------
        x - an array of x coordinates
        y - an array of y coordinates
        buffer - the buffer to add to the resulting polygons
        DTFE - boolean indicating which metric to use
        num_stdev - number of standard deviations to use when setting the metric threshold
        
    
    Fields Populated by run():
    -------------------------
        DLTrace Object with:
            traces - geopandas dataframe with the polygons
            alpha_trace - geopandas dataframe with the alpha_shape polygons
            df - pandas dataframe with the points and metrics
            tri - the Delaunay Tessellation
            dtfe_th - the dtfe threshold (float)
            distance_th - the distance threshold (float)
            d - numpy array of distance values indexed the same as tri.simplices
    """
    
    def __init__(self, x, y, buffer=1.0, DTFE=False, num_stdev=1.0):
        """The constructor requires a set of x and y coordinates.
        The units of x and y do not matter, but the buffer should be
        set to something reasonable given the units of the coordinates.
        The DTFE boolean allows the user to switch from using longest
        edge to using the Delaunay Tessellation Field Estimator (DTFE)
        as a metric for determining which points to retain."""
        
        #Initialize the object
        self.x=x
        self.y=y
        self.buffer=buffer
        self.dtfe=DTFE
        self.num_stdev=num_stdev
        
        #Initialize variables
        self.tri = None
        self.traces = None
        self.dtfe_th = None
        self.distance_th = None
        self.d = None
        self.df = pd.DataFrame({'X':self.x, 'Y':self.y})
        self.alpha_trace = None
        
    #end __init__           
        

    #====================== FUNCTIONS ==========================#
    

    def get_length(self, point1, point2):
        """Compute the distance between 2 points."""
        d=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
        return d
    
    #end get_length
    
    
    def area_triangle(self, points):
        """The area of a triangle computed from 3 points (3x2 array)."""
        return(0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) -
                            np.dot(points[:, 1], np.roll(points[:, 0], 1))))
    
    #end area_triangle
    
    
    def area_delaunay(self,inputs):
        """A helper function that facilitates multithreading or loops for 
        computing the area of all triangles in a Delaunay graph.
        The inputs argument is assumed to be a graph and a scalar tuple."""
        tri, num = inputs
        a = self.area_triangle(tri.points[tri.simplices[num]])
        return(a)
    
    #end area_delaunay
    
    
    def tri_delaunay(self):
        """A function for creating a delaunay graph."""
        tab=np.vstack((self.x,self.y)).T
        tri=spatial.Delaunay(tab)
        return tri
    
    #end tri_delaunay
    
    
    def calculate_dtfe(self):
        """Calculates the area of each of the simplices, converts
        to dtfe and stores in the df data frame as DTFE."""
        nums=np.arange(self.tri.simplices.shape[0])
        ars=np.array([self.area_delaunay((self.tri, n)) for n in nums])
        Ns={}
        As={}
        for s in nums:
            for p in self.tri.simplices[s]:
                if(p in Ns.keys()):
                    Ns[p]+=1
                    As[p]+=ars[s]
                else:
                    Ns[p]=1
                    As[p]=ars[s]
                #end if
            #end for
        #end for
        dd=pd.DataFrame({'Keys':Ns.keys(), 'N':Ns.values(), 'A':As.values()})
        dd=dd.sort_values('Keys')
        self.df['N']=dd.N
        self.df['A']=dd.A
        self.df['DTFE']=np.array(dd.N)/np.array(dd.A)
        
    #end calculate_dtfe
    
    
    def calculate_distance(self):
        """Each point is assigned a distance corresponding to the 
        maximum edge length of all simplices for which that point 
        is a member."""
        D={}
        d=[]
        for simp in self.tri.simplices:
            d1=self.get_length(self.tri.points[simp[0]], self.tri.points[simp[1]])
            d2=self.get_length(self.tri.points[simp[0]], self.tri.points[simp[2]])
            d3=self.get_length(self.tri.points[simp[1]], self.tri.points[simp[2]])
            d.append(np.max([d1,d2,d3]))
        d=np.array(d)
        self.d = d
  
        for n in range(len(self.tri.simplices)):
            for p in self.tri.simplices[n]:
                if p in D.keys():
                    D[p]=np.max([d[n], D[p]])
                else:
                    D[p]=d[n]
                #end if
            #end for
        #end for  
        dd=pd.DataFrame({'Keys':D.keys(), 'DIST':D.values()})
        dd=dd.sort_values('Keys')
        self.df['DIST']=dd.DIST
        
    #end calculate_distance()
    
    
    def dtfe_threshold(self, n=1.0):
        """Use Gaussian Mixtures to finde the DTFE threshold using
        the highest mean plus n standard deviations."""
        LD=np.array(np.log(self.df.DTFE))
        LD=LD.reshape(-1, 1)
        gmm=GaussianMixture(2).fit(LD)
        j=np.where(gmm.means_==np.max(gmm.means_))[0][0]
        m=gmm.means_[j][0]
        sdv=gmm.covariances_[j, 0,0]
        self.dtfe_th=np.exp(m+n*sdv)
        
    #end dtfe_threshold()
    
    
    def distance_threshold(self, n=1.0):
        """Use Gaussian Mixtures to find the distance threshold
        using the lowest mean minus n standard deviations."""
        LD=np.array(np.log(self.df.DIST))
        LD=LD.reshape(-1, 1)
        gmm=GaussianMixture(2).fit(LD)
        j=np.where(gmm.means_==np.min(gmm.means_))[0][0]
        m=gmm.means_[j][0]
        sdv=gmm.covariances_[j, 0,0]
        self.distance_th=np.exp(m-n*sdv)
        
    #end distance_threshold 
    
    
    def get_traces(self):
        """Using the threshold, determine which simplices to retain
        and buffer to create the traces."""
        
        def add_edge(edges, edge_pts, coords, i, j):
            """Adds an edge to the set edges and the cooresponding
            points to the list edge_pts."""
            if (i, j) in edges or (j, i) in edges:
                return
            edges.add( (i, j) )
            edge_pts.append(coords[ [i, j] ])
        #end add_edge
        
        edges=set()
        edge_pts=[]
        if(self.dtfe):
            for s in self.tri.simplices:
                if(self.df.DTFE[s].max()>self.dtfe_th):
                    add_edge(edges, edge_pts, self.tri.points, s[0], s[1])
                    add_edge(edges, edge_pts, self.tri.points, s[0], s[2])
                    add_edge(edges, edge_pts, self.tri.points, s[1], s[2])
                #end if
            #end for
        else:
            simp=self.tri.simplices[self.d < self.distance_th]
            for s in simp:
                add_edge(edges, edge_pts, self.tri.points, s[0], s[1])
                add_edge(edges, edge_pts, self.tri.points, s[0], s[2])
                add_edge(edges, edge_pts, self.tri.points, s[1], s[2])
            #end for
        #end if-else
        
        m=MultiLineString(edge_pts)
        geom = list(polygonize(m))
        for g in range(len(geom)):
            geom[g]=geom[g].buffer(self.buffer)            
        self.traces=gp.GeoDataFrame({'geometry':unary_union(geom)})
        
    #end get_traces
    
    
    def alpha_shape(self, alpha):
        """Borrowed from D. Frezza. Adds a simplex to the polygon
        if the circum radius is < 1/alpha."""
        
        def add_edge(edges, edge_points, coords, i, j):
            if (i, j) in edges or (j, i) in edges:
               # already added
               return
            edges.add( (i, j) )
            edge_points.append(coords[ [i, j] ])
        #end add_edge
        
        coords = np.vstack((self.x,self.y)).T
        self.tri = spatial.Delaunay(coords)
        edges = set()
        edge_points = []
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the
        # triangle
        for ia, ib, ic in self.tri.vertices:
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
        m = MultiLineString(edge_points)
        geom = list(polygonize(m))
        for g in range(len(geom)):
            geom[g]=geom[g].buffer(self.buffer)            
        self.alpha_trace=gp.GeoDataFrame({'geometry':unary_union(geom)})
    
    #end alpha_shape
    
        
    def run(self):
        """The main function that performs the Delaunay tessellation
        and creates the traces and data frame of results."""
        
        self.tri = self.tri_delaunay()
        if(self.dtfe):
            self.calculate_dtfe()
            self.dtfe_threshold(n=self.num_stdev)
            self.get_traces()
        else:
            self.calculate_distance()
            self.distance_threshold(n=self.num_stdev)
            self.get_traces()
        #end if-else
    #end run
    
#end class DLTrace
        
        
        
        
        
        