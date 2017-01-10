"""
Heuristic for high DBVI clustering
@authors: Baptiste R. & Vincent F. & Antoine C.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
import scipy as sp
import sklearn as sklearn
from scipy import spatial
import random as random
import numpy as np
import pandas as pd 
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import copy
import dbvi_computation

###############################################################################
# Parameters
N=400 # number of points in test dataset
k=10  # number of closest neighbors considered to compute core distances initially
d=2   # feature space dimension
###############################################################################


###############################################################################
# Generate a random data set in R2 (disregard the class)
# Change features, informative etc. to change distribution of data

###############################################################################
def rand_ball_gen(radius,center):
    phi = random.uniform(0,2*np.pi)
    r = random.uniform(0,radius)
    
    x = r*np.cos(phi) + center[0]
    y = r*np.sin(phi) + center[1]

    return [x,y]
    
c1 = np.array([0,0])
c2 = np.array([10,10])
c3 = np.array([6,5])
c4 = np.array([-5,5])

num_noise = int(N/20)
#num_noise = 0

A = [rand_ball_gen(2,c1) for i in range(int((N-num_noise)/4))]
B = [rand_ball_gen(2,c2) for i in range(int((N-num_noise)/4))]
C = [rand_ball_gen(2,c3) for i in range(int((N-num_noise)/4))]
D = [rand_ball_gen(2,c4) for i in range(int((N-num_noise)/4))]

noise = [[np.random.uniform(-10, 12.5), np.random.uniform(-10, 12.5)] for i in range(num_noise)]

X = A + B + C + D + noise
random.shuffle(X)
X = np.array(X)

plt.figure(figsize=(20, 14))
plt.scatter(X[:, 0], X[:, 1], marker='o',s=80, alpha = 0.5) # add ", c=Y" to get class
plt.savefig('balls_.png')
###############################################################################


###############################################################################
# Compute core distances

# Get knn distance as in paper
def getcoredist(dataset, testInstance, k):
    distances = []
    for x in range(len(dataset)):
        dist = sp.spatial.distance.euclidean(testInstance, dataset[x])
        distances.append((dataset[x], dist))
        distances = sorted(distances, key = lambda x: (x[1]))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    total = 0
    for elem in neighbors:
        if elem != testInstance:
            total += (1/sp.spatial.distance.euclidean(testInstance,elem))**d
            core_dist = (total/(k-1))**(float(-1)/float(d))
    return core_dist
 

# Compute core distance for all points
dataset = X.tolist()
all_data=pd.DataFrame(dataset) #this data frame will hold core distance
all_data.rename(columns={0: 'x1', 1: 'x2'}, inplace=True)

all_core_dist = list()
for points in dataset:
    core_dist = getcoredist(dataset, points, k)
    all_core_dist.append(core_dist)

all_data['core_dist'] = all_core_dist
###############################################################################


###############################################################################
# Compute MR distance between all points in data set
MR_matrix = np.zeros(shape=(N,N))

for i in range(0,all_data.shape[0]):
    for j in range(0,i+1):
        
        a=list(all_data.iloc[i])[:2]
        b=list(all_data.iloc[j])[:2]
        euclid_i_j = sp.spatial.distance.euclidean(a, b)
        
        mr_dist = max(all_data.iloc[i][2], all_data.iloc[j][2], euclid_i_j)
        
        MR_matrix[i,j]=mr_dist
        MR_matrix[j,i]=mr_dist
###############################################################################
        

###############################################################################
# Create the minimum spanning tree.
mst = minimum_spanning_tree(csr_matrix(MR_matrix))
mst = mst.toarray() # The MST matrix should be symmetric but scipy only keeps one value

# Plot
#G = nx.from_numpy_matrix(mst)
#plt.figure(figsize=(10, 10))
#nx.draw(G, with_labels = True) # Does not respect distances

# Turn mst into a symmetric matrix
for i in range(0,mst.shape[0]):
    for j in range(0,mst.shape[0]):
        if mst[i][j] != 0:
            mst[j][i] = mst[i][j]
###############################################################################


###############################################################################
# Get clusters out of the MST matrix
def get_clusters(A):
    
    G = nx.from_numpy_matrix(A)
    Subgraphs = list(nx.connected_component_subgraphs(G))
    
    all_clusters = list()
    for a_graph in Subgraphs:
        new_cluster = list()    
        for elem in a_graph:
            new_cluster.append(elem)
        all_clusters.append(new_cluster)
    
    return all_clusters
###############################################################################


###############################################################################
# Report initial clusters to main data set (single initial cluster)
all_clusters = get_clusters(mst)

all_data['cluster'] = np.zeros(N)
cluster_number = 0
all_data['color'] = 0 # delete in final code

for a_cluster in all_clusters:    
    for point in a_cluster:
        all_data['cluster'][point] = cluster_number
    cluster_number += 1
###############################################################################


###############################################################################
# create copy of mst and remove single leaf, i.e. elements with one value per line
def create_mst_int(mst):
    A = copy.deepcopy(mst)
    leafs = []
    # 1. identify leafs
    for i in range(0,A.shape[0]):
        if len(np.nonzero(A[i,])[0]) == 1:
            row = i
            col = int(np.nonzero(A[i,])[0])
            leafs.append([row,col])
            
    # 2. remove leafs
    for l in leafs:
        A[l[0]][l[1]] = 0
        A[l[1]][l[0]] = 0
    return A
###############################################################################


###############################################################################
# Return largest edge from (interior) MST
def heavy_edge(mst, protected):
    A = copy.deepcopy(mst)
    maximum = 0
    max_i = 0
    max_j = 0
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[0]):
            if(A[i][j] > maximum) and (i,j) not in protected:
                maximum = A[i][j]
                max_i = i
                max_j = j
    return max_i, max_j
###############################################################################
    
    

###############################################################################
# Reduce tree
DBVI_list = []
protected = []
DBVI_history = []

p=0
end=0
while (p<20 and end==0):
    
    interior_mst = create_mst_int(mst)                            #get interior mst
    max_i, max_j = heavy_edge(interior_mst, protected)            #find heaviest edge in interior mst  
    
    old_value = mst[max_i][max_j] #keep it in case
    old_all_data = copy.deepcopy(all_data) #keep it in case
    
    mst[max_i][max_j], mst[max_j][max_i] = 0, 0                   #delete the edge in global mst
    
    
    all_clusters = get_clusters(mst)                              #extract clusters
    cluster_number = 0    
    for a_cluster in all_clusters:    
        for point in a_cluster:
            all_data['cluster'][point] = cluster_number
        cluster_number += 1
    #print(all_data)   
    
    
    DBVIndex,DBVI = dbvi_computation3.DBVI_comp(all_data)         #compute dbvi
    DBVI_list.append(DBVIndex)
    DBVI_history.append(DBVI)

    for clus in DBVI:
        if p > 0 and DBVI_list[p-1] > DBVI_list[p]:               #stop when DVBI decreases
            end = 1 
            plt.figure(figsize=(20, 14))
            plt.scatter(all_data['x1'], all_data['x2'],c=all_data['cluster'], s=80, alpha=0.5)
            plt.savefig("line_%s"%p)
            all_data = old_all_data
            STOP_data = (DBVI, DBVIndex)
            break
    
    if end == 0:        
        plt.figure(figsize=(20, 14))
        plt.scatter(all_data['x1'], all_data['x2'],c=all_data['cluster'], s=80, alpha=0.5)
        plt.savefig("line_%s"%p)
        plt.show()

    p+=1
###############################################################################

# print results and simulation parameters 
for t in range(0,len(DBVI_history)):
    print("*** STEP %s ***"%t)
    for clus in DBVI_history[t].keys():
        print("DSC %s = %s"%(clus,DBVI_history[t][clus][2]))
        print("# Internal nodes %s = %s"%(clus,len(DBVI_history[t][clus][4])))        
        print("DSPC %s = %s"%(clus,DBVI_history[t][clus][5]))
        print("VCI %s = %s"%(clus,DBVI_history[t][clus][6]))
        DBVI_history[t][clus][5].sort()
        print("DSC/min(DSPC) %s = %s"%(clus,float(DBVI_history[t][clus][2])/float(DBVI_history[t][clus][5][1])))
        print("")   
    print("--> DBVI = %s"%DBVI_list[t])
    print("")
    print("")
