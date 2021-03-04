## Matrix Decompositions in Data Analysis
## Spring semester 2021
##
## Assignment 1
## Due 2 February 2021 at 23:59

## Student's information
## Name:
## Email:
## Student ID #:

##
## This file contains stubs of code to help you to do your 
## assignment. You can fill your parts at the indicated positions
## and return this file as a part of your solution. Remember that
## if you define any functions etc. in other files, you must include 
## those files, as well. You do not have to include standard libraries
## or any files that we provide.
##
## Remember to fill your name and matriculation number above.
##

## Preamble
###########

## This file is intended to be used with python3. You can run it with
## python3 assignment1.py
## though as is, it won't run (missing matrix D from Task 3).
##
## We use numpy, scikit-learn, matplotlib, cartopy and networkx packages.
## You can install them using pip or anaconda if you don't have them. We
## import them here so we get import errors as soon as possible.
##
## Cartopy requires external libraries; if you have issues installing it,
## read https://scitools.org.uk/cartopy/docs/latest/installing.html .
## If you cannot install cartopy, you can leave it out, remove all
## "projection=..." parameters from plt.axes() calls and the ax.set_global()
## and ax.coastlines() calls and just use latitude and longitude as x and y
## coordinates. You will loose the coastlines and you'll notice that the
## projection is off, though. 
import numpy as np
import scipy
from numpy.linalg import svd
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite

## Task 1
#########

## Load the data. We could use pandas to get tables, but as we're just doing
## matrix stuff numpy arrays are fine. 
data = np.genfromtxt('worldclim.csv', delimiter=',', skip_header=1)
coord = np.genfromtxt('coordinates.csv', delimiter=',', skip_header=1)
lat = coord[:,0]
lon = coord[:,1]

## Familiarize yourself with the data; do not include that part to your
## report.

## Compute SVD of data
U, S, V = svd(data, full_matrices=False)

## Plot the base map; nothing is shown yet
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
## If you want to see how it looks, write
#plt.show()

## Plot the first column of U so that the color indicates the value
plt.scatter(lat, lon, s=1, c = U[:,0])
plt.show()

## Another plot with different colormap and colorbar
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
plt.scatter(lat, lon, s=1, c = U[:,0], cmap='RdYlBu')
plt.colorbar(shrink=0.5)
plt.show()

## Plot the second column
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
plt.scatter(lat, lon, s=1, c = U[:,1], cmap='RdYlBu')
plt.colorbar(shrink=0.5)
plt.show()

## YOUR PART STARTS HERE


## YOUR PART ENDS HERE

#############
## Task 2
#############

## You have to implement the rank selection techniques yourself. 
## Remember to use the normalized data.

## If S is an array that contains the singular values, you can plot them by
plt.plot(S)
plt.show()

## YOUR PART STARTS HERE


## YOUR PART ENDS HERE

#############
## Task 3
#############

## We again use the normalized data. If that is contained in matrix 'D'
## we can compute the k-means to 5 clusters with 10 re-starts as
clustering = KMeans(n_clusters=5, n_init=10).fit(D)
idx = clustering.labels_

## The plotting is as in Task 1
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
plt.scatter(lat, lon, s=1, c = idx, cmap='tab10')
plt.show()

## To compare two different clusterings, we must match their labels.
## Below is a function that computes the maximum matching between two
## clustering labels and returns the matching labels for the second
## argument.

def matching(idxA, idxB):
    labelsA = np.unique(idxA)
    m = len(labelsA)
    labelsB = np.unique(idxB)
    n = len(labelsB)
    W = np.zeros((m, n))
    for j in range(n):
        for i in range(m):
            W[i, j] = -sum(idxA[idxB == labelsB[j]] == labelsA[i])
    G = bipartite.from_biadjacency_matrix(scipy.sparse.coo_matrix(W))
    top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    match = bipartite.minimum_weight_full_matching(G, top_nodes)
    new_labels = dict()
    for i in range(n):
        new_labels[labelsB[i]] = labelsA[match[i+n]]
    matched = np.array([new_labels[idxB[i]] for i in range(len(idxB))])
    return matched

## Here's how to use it. This clustering is only shown here to explain how to use the matching algorithm. Do not use these in actual report!
#clustering1 = KMeans(n_clusters=5, n_init=1, init='random').fit(data)
#idx1 = clustering1.labels_
#clustering2 = KMeans(n_clusters=5, n_init=1, init='random').fit(data)
#idx2 = clustering2.labels_

#ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
#ax1.set_global()
#ax1.coastlines()
#ax1.set_title('First clustering')
#ax1.scatter(lat, lon, s=1, c = idx1, cmap='tab10')
#ax2 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
#ax2.set_global()
#ax2.coastlines()
#ax2.set_title('Second clustering, orig. labels')
#ax2.scatter(lat, lon, s=1, c = idx2, cmap='tab10')
#ax3 = plt.subplot(1, 3, 3, projection=ccrs.PlateCarree())
#ax3.set_global()
#ax3.coastlines()
#ax3.set_title('Second clustering, matched labels')
#ax3.scatter(lat, lon, s=1, c = matching(idx1, idx2), cmap='tab10')
#plt.show()

## YOUR PART STARTS HERE

## YOUR PART ENDS HERE
