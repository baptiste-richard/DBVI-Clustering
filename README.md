# DBVI-Clustering Heuristic
Authors: Baptiste R. - Vincent F. - Antoine C.

The Density Based Validation Index (DBVI), as developed by Davoud Moulavi Et al [1], provides a good way of measuring the quality of a clustering based on within-cluster density and on between-cluster density connectedness. Through a newly defined kernel density function, such index efficiently assesses the quality of a given clustering, independently of whether or not the clusters are globular.

Though such method allows to accurately rate a given clustering solution, the paper does not provide an algorithm allowing to find such a clustering. We therefore present a clustering algorithm, inspired from the DBVI methodology, that relies on the density sparseness inside a given cluster as well as on the separation between distinct clusters, in the mutual reachability space.

Interestingly, this heuristic is basically non-parametric, as it requires no assumptions regarding the number of clusters for instance. Moreover, the resulting clustering seems to not only be the ground truth clustering, but to also coincide with the very iteration of the algorithm where the DBVI value is maximized. Such results call for further theoretical research regarding the convexity of the DBVI values as the algorithm iterates.

Implementation of DBVI clustering method --> Density-Based Clustering Validation

http://epubs.siam.org/doi/pdf/10.1137/1.9781611973440.96 [1]
