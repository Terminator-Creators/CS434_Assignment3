import numpy as np
import random
from statistics import mode

class Node():
	"""
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree


class DecisionTreeClassifier():
	"""
	Decision Tree Classifier. Class for building the decision tree and making predictions

	Parameters:
	------------
	max_depth: int
		The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
	"""

	def __init__(self, max_depth=None, max_features=0):
		self.max_depth = max_depth
		self.max_features = max_features
		
	# take in features X and labels y
	# build a tree
	def fit(self, X, y):
		self.num_classes = len(set(y))
		self.root = self.build_tree(X, y, depth=1)
		

	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree
			else:
				node = node.right_tree
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	# function to build a decision tree
	def build_tree(self, X, y, depth):
		num_samples, num_features = X.shape
		# which features we are considering for splitting on
		self.features_idx = np.arange(0, X.shape[1])

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_gain = 0.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None

		# what we would predict at this node if we had to
		# majority class
		num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
		prediction = np.argmax(num_samples_per_class)
		
		# Added code to let RandomForest bag the features
		if(self.max_features):
			max_features = []
			temp_feat = list(self.features_idx)
			
			for i in range(self.max_features):
				rand = random.randint(0,len(temp_feat)-1)
				max_features.append(temp_feat[rand])
				temp_feat.pop(rand)

		# If no max_features were specified when creating this tree, we don't subsample
		else:
			max_features = self.features_idx

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature
			for feature in max_features:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
					# if we have a better gain, use this split and keep track of data
					if gain > best_gain:
						best_gain = gain
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
		
		# if we haven't hit a leaf node
		# add subtrees recursively
		if best_gain > 0.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
			right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]


		# calculate gini impurity and gain for y, left_y, right_y
		gain = self.calculate_gini_gain(y, left_y, right_y)
		return gain, left_X, right_X, left_y, right_y

	def calculate_gini_gain(self, y, left_y, right_y):
		# not a leaf node
		# calculate gini impurity and gain
		gain = 0
		if len(left_y) > 0 and len(right_y) > 0:

			########################################
			#       YOUR CODE GOES HERE            #
			########################################
			left_yes = 0
			left_no = 0
			right_yes = 0
			right_no = 0

			for i in left_y:
				if i == 0:
					left_no += 1
				else:
					left_yes += 1

			for i in right_y:
				if i == 0:
					right_no += 1
				else:
					right_yes += 1

			total = len(left_y) + len(right_y)
			#calculate all impurities and return gain
			gini_left = 1 - np.square(left_yes/(left_yes+left_no)) - np.square(left_no/(left_yes+left_no))
			gini_right = 1 - np.square(right_yes/(right_yes+right_no)) - np.square(right_no/(right_yes+right_no))
			gain = ((left_yes+left_no)/total) * gini_left + ((right_yes + right_no)/total) * gini_right

			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0

		  
class RandomForestClassifier():
	"""
	Random Forest Classifier. Build a forest of decision trees.
	Use this forest for ensemble predictions

	YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

	Parameters:
	-----------
	n_trees: int
		Number of trees in forest/ensemble
	max_features: int
		Maximum number of features to consider for a split when feature bagging
	max_depth: int
		Maximum depth of any decision tree in forest/ensemble
	"""
	def __init__(self, n_trees, max_features, max_depth):
		self.n_trees = n_trees
		self.max_features = max_features
		self.max_depth = max_depth

	
	# fit all trees
	def fit(self, X, y):
		print('Fitting Random Forest...\n')
		self.root = []
		self.num_classes = []
		for i in range(self.n_trees):
			bagged_X, bagged_y = self.bag_data(X, y)
			print(i+1, end='\t\r')

			# Essentially the fit function from DecisionTreeClassifier
			tree = DecisionTreeClassifier(self.max_depth, self.max_features)
			tree.num_classes = len(set(bagged_y))
			self.num_classes.append(len(set(bagged_y)))
			self.root.append(tree.build_tree(bagged_X, bagged_y, 0))
	
		print("Fitted " + str(self.n_trees) + " Trees.\n")

	# This is supposed to be called for each tree, not only once
	def bag_data(self, X, y, proportion=1.0):
		bagged_X = []
		bagged_y = []
		for j in range(2098):
			rand = random.randint(0, 2097)
			bagged_X.append(X[rand])
			bagged_y.append(y[rand])
		# ensure data is still numpy arrays
		return np.array(bagged_X), np.array(bagged_y)

	# Prediction function that uses helper
	def predict(self, X):
		return [self._predict(x) for x in X]
	
	# Prediction helper that runs each prediction for each tree and return the mode
	def _predict(self, x):
		preds = []
		for i in range(self.n_trees):
			node = self.root[i]
			while node.left_tree:
				if x[node.feature] < node.split:
					node = node.left_tree
				else:
					node = node.right_tree
			preds.append(node.prediction)
		return mode(preds)


################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################
class AdaBoostClassifier():
	"""
	Initialization:
	Max depth is set to 1.
	"""
	def __init__(self, max_depth=1):
		self.max_depth = max_depth
	# take in features X and labels y
	# build a tree
	def fit(self, X, y, d):
		self.root = self.build_tree(X, y, d, depth=1)
		return self.root

	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		if (example[node.feature] >= node.split):
			# print("left: ", node.left_tree)
			return node.left_tree
		else:
			# print("right: ", node.right_tree)
			return node.right_tree

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	# function to build a decision tree
	def build_tree(self, X, y, d, depth):
		# num_samples, num_features = X.shape
		# which features we are considering for splitting on
		self.features_idx = np.arange(0, X.shape[1])

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_gain = 10000.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None

		# what we would predict at this node if we had to
		# majority class
		num_samples_per_class = [np.sum(y == i) for i in [-1, 1]]
		prediction = np.argmax(num_samples_per_class)

		# # if we haven't hit the maximum depth, keep building
		# if depth <= self.max_depth:
		# 	# consider each feature
		for feature in self.features_idx:
			# consider the set of all values for that feature to split on
			possible_splits = np.unique(X[:, feature])
			for split in possible_splits:
				error1 = 0
				error2 = 0
				for x in range(len(X)):
					if (X[x][feature] < split):
						if(y[x] == 1):
							error1+=d[x]
						else:
							error2+=d[x]
					if (X[x][feature] >= split): 
						if(y[x]==1):
							error2+=d[x]
						else:
							error1+=d[x]
				if (error1 < best_gain):
					best_gain = error1
					best_feature = feature
					best_split = split
					best_left_y = -1
					best_right_y = 1
				if(error2 < best_gain):
					best_gain = error2
					best_feature = feature
					best_split = split
					best_left_y = 1
					best_right_y = -1

		# print(best_feature, best_split, best_left_y, best_right_y)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=best_left_y, right_tree=best_right_y)









