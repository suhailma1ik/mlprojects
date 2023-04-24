class Node:
    def __init__(self, value):
        self.value = value
        self.children = {}

class DecisionTree:
    def __init__(self):
        self.root = None
    
    def train(self, data, target_attr):
        # data is a list of dictionaries where each dictionary represents an instance
        # target_attr is the name of the attribute that we are trying to predict
        self.root = self.build_tree(data, target_attr)
    
    def classify(self, instance):
        node = self.root
        while node.children:
            value = instance[node.value]
            node = node.children[value]
        return node.value
    
    def build_tree(self, data, target_attr):
        # If all instances have the same target attribute value, return a leaf node with that value
        if all(instance[target_attr] == data[0][target_attr] for instance in data):
            return Node(data[0][target_attr])
        
        # If there are no more attributes to split on, return a leaf node with the most common target attribute value
        if len(data[0]) == 1:
            target_values = [instance[target_attr] for instance in data]
            most_common = max(set(target_values), key=target_values.count)
            return Node(most_common)
        
        # Otherwise, choose the best attribute to split on and create a new node for each possible value
        best_attr = self.get_best_attr(data, target_attr)
        node = Node(best_attr)
        for value in set(instance[best_attr] for instance in data):
            subset = [instance for instance in data if instance[best_attr] == value]
            child_node = self.build_tree(subset, target_attr)
            node.children[value] = child_node
        return node
    
    def get_best_attr(self, data, target_attr):
        # Compute the information gain for each attribute and choose the one with the highest gain
        total_entropy = self.entropy(data, target_attr)
        best_attr, best_gain = None, 0
        for attr in data[0]:
            if attr == target_attr:
                continue
            gain = total_entropy - self.split_entropy(data, attr, target_attr)
            if gain > best_gain:
                best_attr, best_gain = attr, gain
        return best_attr
    
    def entropy(self, data, target_attr):
        # Compute the entropy of the target attribute in the given data
        target_values = [instance[target_attr] for instance in data]
        value_counts = {value: target_values.count(value) for value in set(target_values)}
        probs = [count / len(data) for count in value_counts.values()]
        return -sum(p * math.log2(p) for p in probs)
    
    def split_entropy(self, data, attr, target_attr):
        # Compute the entropy of the target attribute
