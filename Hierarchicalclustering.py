import numpy as np
from scipy.spatial.distance import pdist, squareform

def agglomerative_clustering(data, k):
    n = len(data)
    
    # Compute the pairwise distances between data points
    pairwise_dist = squareform(pdist(data))
    
    # Initialize each data point as a separate cluster
    clusters = [[i] for i in range(n)]
    
    # Perform agglomerative clustering until k clusters are left
    while len(clusters) > k:
        # Find the two closest clusters based on single-linkage distance
        min_dist = np.inf
        merge_index = None
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = np.min(pairwise_dist[clusters[i], clusters[j]])
                if dist < min_dist:
                    min_dist = dist
                    merge_index = (i, j)
        
        # Merge the two closest clusters
        i, j = merge_index
        clusters[i] += clusters[j]
        del clusters[j]
    
    # Return the cluster assignments for each data point
    cluster_assignments = np.zeros(n)
    for i, cluster in enumerate(clusters):
        cluster_assignments[cluster] = i
    
    return cluster_assignments


# implement

# Generate some sample data
np.random.seed(0)
data = np.random.rand(20, 2)

# Perform agglomerative clustering with k=3 clusters
cluster_assignments = agglomerative_clustering(data, k=3)

# Print the cluster assignments
print(cluster_assignments)
