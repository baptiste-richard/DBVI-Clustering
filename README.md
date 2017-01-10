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

The goal of this heuristic is to proceed to sequential removals of some targeted edges of MST(G). Such operation will yield a disconnected graph, whose fully connected sub-graphs will correspond to the final clusters. We wish to repeat the above operation several time so as to gradually improve the quality of the resulting clustering.

The heuristic we present sequentially removes some of the MST heaviest edges so as to create clusters that have low density sparseness, and high density separation with other clusters. Empirically, it turned out that the ground truth clustering seems to be reached whenever the DBVI value of the current clustering is maximized. The algorithm therefore uses this fact as stopping condition.

Implementation of DBVI clustering method --> Density-Based Clustering Validation

http://epubs.siam.org/doi/pdf/10.1137/1.9781611973440.96 [1]
