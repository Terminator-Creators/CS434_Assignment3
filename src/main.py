import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set()
random.seed()

import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info
from tree import DecisionTreeClassifier, RandomForestClassifier

def load_args():

	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--county_dict', default=1, type=int)
	parser.add_argument('--decision_tree', default=1, type=int)
	parser.add_argument('--random_forest', default=1, type=int)
	parser.add_argument('--ada_boost', default=1, type=int)
	parser.add_argument('--root_dir', default='../data/', type=str)
	args = parser.parse_args()
 
	return args


def county_info(args):
	county_dict = load_dictionary(args.root_dir)
	dictionary_info(county_dict)

def decision_tree_testing(x_train, y_train, x_test, y_test):
	print('Decision Tree\n\n')
	clf = DecisionTreeClassifier(max_depth=20)
	clf.fit(x_train, y_train)
	preds_train = clf.predict(x_train)
	preds_test = clf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = clf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))

def create_trees(x_train, y_train, x_test, y_test, maxdepth):
	#print('Decision Tree\n\n')
	clf = DecisionTreeClassifier(max_depth=maxdepth)
	clf.fit(x_train, y_train)
	preds_train = clf.predict(x_train)
	preds_test = clf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	#print('Train {}'.format(train_accuracy))
	#print('Test {}'.format(test_accuracy))
	preds = clf.predict(x_test)
	#print('F1 Test {}'.format(f1(y_test, preds)))
	return (f1(y_test, preds)),train_accuracy, test_accuracy


def random_forest_testing(x_train, y_train, x_test, y_test):
	print('Random Forest Loop\n\n')
	train_list = []
	test_list = []
	F1_list = []
	
	for i in range(10,210,10):
		rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=i)
		rclf.fit(x_train, y_train)
		preds_train = rclf.predict(x_train)
		preds_test = rclf.predict(x_test)
		train_accuracy = accuracy_score(preds_train, y_train)
		test_accuracy = accuracy_score(preds_test, y_test)
		print('Train {}'.format(train_accuracy))
		print('Test {}'.format(test_accuracy))
		preds = rclf.predict(x_test)
		print('F1 Test {}'.format(f1(y_test, preds)))

		# Grab the useful number per cycle
		train_list.append(train_accuracy)
		test_list.append(test_accuracy)
		F1_list.append(f1(y_test, preds))
	
	plt.rcParams['font.family'] = ['serif']
	x = range(10,210,10)
	plt.plot(x, train_list, x, test_list, x, F1_list)
	plt.xlabel("n_trees")
	plt.xticks(x)
	plt.ylabel("Accuracies")
	plt.legend()
	plt.savefig("RandomForest.png")
	plt.clf()



def OG_random_forest_testing(x_train, y_train, x_test, y_test):
	print('Random Forest\n\n')
	rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=50)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))
	
def create_trees_wrapper():
	f1_accuracy = [0 for f1_accuracy in range(25)]
	test_accuracy = [0 for test_accuracy in range(25)]
	train_accuracy = [0 for train_accuracy in range(25)]
	for x in range(25):
		f1_accuracy[x], train_accuracy[x], test_accuracy[x]= create_trees(x_train, y_train, x_test, y_test, x+1)
		
	# Plot the training and testing ASE's vs d
	d = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
	plt.rcParams['font.family'] = ['serif']

	plt.plot(d, train_accuracy)
	plt.xlabel("Max Depth")
	plt.xticks([1,5,10,15,20,25])
	plt.ylabel("Training Accuracy")
	plt.savefig("train.png")
	plt.clf()

	# Plot the training and testing ASE's vs d
	d = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
	plt.rcParams['font.family'] = ['serif']

	plt.plot(d,test_accuracy)
	plt.xlabel("Max Depth")
	plt.xticks([1,5,10,15,20,25])
	plt.ylabel("Testing Accuracy")
	plt.savefig("test.png")
	plt.clf()

	# Plot the training and testing ASE's vs d
	d = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
	plt.rcParams['font.family'] = ['serif']

	plt.plot(d,f1_accuracy)
	plt.xlabel("Max Depth")
	plt.xticks([1,5,10,15,20,25])
	plt.ylabel("F1 Testing Accuracy")
	plt.savefig("f1test.png")
	plt.clf()


###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
	args = load_args()

	x_train, y_train, x_test, y_test = load_data(args.root_dir)
	if args.county_dict == 1:
		county_info(args)
	if args.decision_tree == 1:
		decision_tree_testing(x_train, y_train, x_test, y_test)
		create_trees_wrapper()
	
	if args.random_forest == 1:
		random_forest_testing(x_train, y_train, x_test, y_test)

	print('Done')
	
	





