import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DecisionTree:
    
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        
    def predict(self, X):
        predictions = []
        for x in X:
            node = self.tree
            while node.get('threshold'):
                if x[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node['class'])
        return np.array(predictions)
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(set(y))
        class_counts = [len(y[y == c]) for c in range(n_classes)]
        majority_class = np.argmax(class_counts)
        
        # stopping conditions
        if n_samples == 0:
            return {'class': majority_class}
        if len(set(y)) == 1:
            return {'class': y[0]}
        if depth == self.max_depth:
            return {'class': majority_class}
        
        # find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        # build left and right subtrees
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth+1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth+1)
        
        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left_tree, 'right': right_tree}
    
    def _information_gain(self, y, left_y, right_y):
        p = len(left_y) / len(y)
        entropy_before = self._entropy(y)
        entropy_after = p * self._entropy(left_y) + (1 - p) * self._entropy(right_y)
        return entropy_before - entropy_after
    
    def _entropy(self, y):
        n_samples = len(y)
        if n_samples == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / n_samples
        return -np.sum([p * np.log2(p) for p in probs if p > 0])
    
    def _find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_info_gain = -1
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            thresholds = sorted(set(X[:, feature]))
            for i in range(1, len(thresholds)):
                threshold = (thresholds[i-1] + thresholds[i]) / 2
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                if len(y[left_indices]) > 0 and len(y[right_indices]) > 0:
                    info_gain = self._information_gain(y, y[left_indices], y[right_indices])
                    if info_gain > best_info_gain:
                        best_feature = feature
                        best_threshold = threshold
                        best_info_gain = info_gain
        
        return best_feature, best_threshold

# implementation

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Predict on test set
y_pred = tree.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
