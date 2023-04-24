import numpy as np

class KMeans:
    def __init__(self, k=2, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        
    def fit(self, X):
        self.centroids = []
        for i in range(self.k):
            self.centroids.append(X[np.random.choice(range(len(X)))])
        
        for i in range(self.max_iterations):
            clusters = [[] for _ in range(self.k)]
            
            for x in X:
                distances = [np.linalg.norm(x-c) for c in self.centroids]
                cluster = np.argmin(distances)
                clusters[cluster].append(x)
            
            prev_centroids = self.centroids.copy()
            
            for idx, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    self.centroids[idx] = X[np.random.choice(range(len(X)))]
                else:
                    self.centroids[idx] = np.mean(cluster, axis=0)
            
            if np.array_equal(prev_centroids, self.centroids):
                break
                
    def predict(self, X):
        clusters = [[] for _ in range(self.k)]
        for x in X:
            distances = [np.linalg.norm(x-c) for c in self.centroids]
            cluster = np.argmin(distances)
            clusters[cluster].append(x)
        return clusters


# how to call class

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

kmeans = KMeans(k=2, max_iterations=100)
kmeans.fit(X)

clusters = kmeans.predict(X)
print(clusters)