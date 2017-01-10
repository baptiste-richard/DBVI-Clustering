# DBVI Clustering Heuristic
Authors: Baptiste R. - Vincent F. - Antoine C.

The Density Based Validation Index (DBVI), as developed by Davoud Moulavi Et al [1], provides a good way of measuring the quality of a clustering based on within-cluster density and on between-cluster density connectedness. Through a newly defined kernel density function, such index efficiently assesses the quality of a given clustering, independently of whether or not the clusters are globular.

Though such method allows to accurately rate a given clustering solution, the paper does not provide an algorithm allowing to find such a clustering. We therefore present a clustering algorithm, inspired from the DBVI methodology, that relies on the density sparseness inside a given cluster as well as on the separation between distinct clusters, in the mutual reachability space.

Interestingly, this heuristic is basically non-parametric, as it requires no assumptions regarding the number of clusters for instance. Moreover, the resulting clustering seems to not only be the ground truth clustering, but to also coincide with the very iteration of the algorithm where the DBVI value is maximized. Such results call for further theoretical research regarding the convexity of the DBVI values as the algorithm iterates.

***

The Clustering Heuristic presented here relies on the following concepts defined in Davoud Moulavi Et al:

- Core Distance
- Mutual Reachability Distance (MR Distance)
- MR Graph
- Minimum Spanning Tree (MST) derived from MR Graphs
- Density Sparseness of a Cluster Ci, denoted DSC(Ci)
- Density Separation of a Pair of Clusters, (Ci,Cj), denoted DSPC(Ci,Cj)
- Validity Index of Cluster Ci, VIC(Ci)
- Density Based Validation Index of Clustering C, denoted DBVI(C)

***

The goal of this heuristic is to proceed to sequential removals of some targeted edges of MST(G). Such operation will yield a disconnected graph, whose fully connected sub-graphs will correspond to the final clusters. We wish to repeat the above operation several time so as to gradually improve the quality of the resulting clustering. Before describing these steps, we make the following remarks regarding the notations we use:

1. After removing an edge from the initial MST(G), the graph is therefore disconnected. However, for notation purposes, we denote that disconnected graph as MST(1)(G).
2. More generally, MST(m+1)(G) is the graph obtained via the removal of the heaviest edge from MST(m)(G).
3. Finally, we notice that the number of clusters at the m−th step is m+1, which is also the number of fully connected sub-graphs of MST(m)(G).
4. Recall that MST(m)(G) can simply be represented as a symmetric matrix, A_MST(m) = {a(m)} where, 
  - aij(m) = aji(m) = 0 if there does not exist an edge between xi and xj in MST(m)(G)
  - aij(m) = aji(m) = MR(xi,xj) if there exists an edge between xi and xj in MST(m)(G)

The heuristic we present here searches the matrix A_MST(m) and sequentially remove some of its heaviest edges, so as to create clusters that have low density sparseness, and high density separation with other clusters. Empirically, it turned out that the ground truth clustering seems to be reached whenever the DBVI value of the current clustering is maximized. The algorithm therefore uses this fact as stopping condition.

***

DBVI-Clustering/heuristic_1.png

DBVI-Clustering/heuristic_2.png

***
Implementation of DBVI clustering method --> Density-Based Clustering Validation

http://epubs.siam.org/doi/pdf/10.1137/1.9781611973440.96 [1]
