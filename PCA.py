import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        # Compute the mean of the input data
        self.mean = np.mean(X, axis=0)
        
        # Center the input data
        X_centered = X - self.mean
        
        # Compute the covariance matrix of the centered data
        cov_matrix = np.cov(X_centered.T)
        
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)
        
        # Sort the eigenvectors in descending order of eigenvalues
        indices = np.argsort(eig_values)[::-1]
        eig_vectors = eig_vectors[:, indices]
        
        # Select the top n_components eigenvectors as the principal components
        self.components = eig_vectors[:, :self.n_components]
    
    def transform(self, X):
        # Project the input data onto the principal components
        X_centered = X - self.mean
        return X_centered.dot(self.components)
# implementation

# Load the input data
X = np.loadtxt('data.txt')

# Apply PCA to the input data
pca = PCA(n_components=2)
pca.fit(X)
X_transformed = pca.transform(X)

# Apply KMeans clustering to the transformed data
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(X_transformed)

print(y_pred)