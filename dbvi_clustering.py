# DBVI Clustering
# 1. Kmeans ++ clustering implementation

import pandas as pd 
import numpy as np 
%matplotlib inline 
import matplotlib as plt 
plt.style.use('ggplot')
import random
from sklearn.cluster import KMeans
import networkx as nx


# using polar coordinates, we define a point generator on a ball with given radius and center
# radius is defined as a positive number
# center is an array of size 3 - cartesian coordinates

#problem dimension - set to R2 feature space
d = 2

def rand_ball_gen(radius,center):
    phi = random.uniform(0,2*np.pi)
    r = random.uniform(0,radius)
    
    x = r*np.cos(phi) + center[0]
    y = r*np.sin(phi) + center[1]

    return np.array([x,y])
    
    
# generate 4 distinct centers and generate dataset using those centers
c1 = np.array([0,0])
c2 = np.array([10,10])
c3 = np.array([-5,-5])
c4 = np.array([-5,5])

A = [rand_ball_gen(10,c1) for i in xrange(100)]
B = [rand_ball_gen(10,c2) for i in xrange(100)]
C = [rand_ball_gen(10,c3) for i in xrange(100)]
D = [rand_ball_gen(10,c4) for i in xrange(100)]
X = A + B + C + D
random.shuffle(X)


# Kmeans ++ algorithm on that data for k = 4 - kmeans ++ initialization

total_iteration = []
inertia = 0

for k in range(1,300):
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init = 1, max_iter = k, random_state = 92).fit(X)
    if inertia == kmeans.inertia_:
        num_iteration = k
        break
    inertia = kmeans.inertia_
total_iteration.append(num_iteration)

print("kmeans ++ ok")
print(str(round(np.mean(total_iteration),2))+" iterations")
print(str(kmeans.inertia_)+" inertia")


# data vizualisation of k-means ++ clustering results
fig = plt.pyplot.figure(num=None, figsize=(10, 10), dpi=200, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

x =[]
y =[]

for i in range(0,len(X)):
    x.append(X[i][0])
    y.append(X[i][1])

color = kmeans.labels_
ax.scatter(x, y, s=50, c=color, marker='o',alpha=0.5)
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=150, linewidths=3,color='k', zorder=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')


# populate clusters resulting from our kmeans++ algorithm using labels
clusters = [[] for _ in range(len(kmeans.cluster_centers_))]

for i in range(0,len(kmeans.labels_)):
    for label in range(0,len(kmeans.cluster_centers_)):
        if kmeans.labels_[i] == label:
            clusters[label].append(X[i])

print(str(len(clusters))+" clusters")
print("***")
for i in range(0,len(clusters)):
    print("cluster "+str(i)+" - "+str(len(clusters[i]))+" points centered on "+str(kmeans.cluster_centers_[i]))
    
    
    
# 2. DBVI Clustering
# Algorithm implementation following Davoud Moulavi - Pablo A. Jaskowiak† - Ricardo J. G. B. - Campello† Arthur Zimek† - Jörg Sander work.

# core distance definition
def core_dist(clusters,num_clus,pos_clus):
    active_cluster = list(clusters[num_clus])
    x = active_cluster.pop(pos_clus)
    knn_sum = 0
    for i in range(0,len(active_cluster)):
        knn_sum += (1/np.sqrt((x[0]-active_cluster[i][0])**2+(x[1]-active_cluster[i][1])**2))**d
    core_dist = (float(knn_sum)/float(len(active_cluster)))**(-1/d)
    return core_dist

# MR distance definition
def mr_distance(clusters,num_clus,pos_clus_x,pos_clus_y):
    active_cluster = list(clusters[num_clus])
    core_dist_x = core_dist(clusters,num_clus,pos_clus_x)
    core_dist_y = core_dist(clusters,num_clus,pos_clus_y)
    euc_dist_xy = np.sqrt((active_cluster[pos_clus_x][0]-active_cluster[pos_clus_y][0])**2+(active_cluster[pos_clus_x][1]-active_cluster[pos_clus_y][1])**2)
    return max(core_dist_x,core_dist_y,euc_dist_xy)


# generate MR Graphs using networkx
# MR graph is undirected, complete - nodes correpond to data points in cluster, weights to MR distance
# Note: we can verify that we get n(n-1)/2 arcs for each complete graph

G0 = nx.Graph()
for i in range(0,len(clusters[0])):
    G0.add_node(i,pos0=(clusters[0][i][0],clusters[0][i][1]))
    pos0=nx.get_node_attributes(G0,'pos0')

for j in range(0,len(clusters[0])):
    for i in range(1+j,len(clusters[0])):
        G0.add_edge(j, i, weight = mr_distance(clusters,0,j,i))
        
# plot one graph to give an example of complete graph
plt.pyplot.figure(figsize=(12,10))   
nx.draw(G0,pos0)

# compute and plot minimum spanning tree
min_tree_0 = nx.minimum_spanning_tree(G0)
plt.pyplot.figure(figsize=(12,10))   
nx.draw(min_tree_0)

# compute Density Sparsness of Cluster as the max weight of our MST
max_weight_0 = 0
ind_max_0 = -1
for i in range(0,len(min_tree_0.edges())):
    if min_tree_0.edges(data=True)[i][2]['weight'] > max_weight_0:
        max_weight_0 = min_tree_0.edges(data=True)[i][2]['weight']
        ind_max_0 = i
DSC_C0 = max_weight_0

