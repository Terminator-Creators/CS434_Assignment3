import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set()
random.seed()

import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info, adaboost_data
from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier

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


def ft_random_forest_testing(x_train, y_train, x_test, y_test):
	print('Random Forest Feature Loop\n\n')
	train_list = []
	test_list = []
	F1_list = []
	
	for i in [1, 2, 5, 8, 10, 20, 25, 35, 50]:
		rclf = RandomForestClassifier(max_depth=7, max_features=i, n_trees=50)
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
	x = [1, 2, 5, 8, 10, 20, 25, 35, 50]
	plt.plot(x, train_list, x, test_list, x, F1_list)
	plt.xlabel("# of max_features")
	plt.xticks(x)
	plt.ylabel("Accuracies")
	plt.legend()
	plt.savefig("RandomForestFeatures.png")
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

def ada_boost_testing(x_train, y_train, x_test, y_test, l=10):
	print('Adaboost\n\n')
	adbt = AdaBoostClassifier()
	L = l
	h = []
	e = np.zeros(L)
	a = np.zeros(L)
	d = np.full((L,2098),1/2098)
	for t in range(L):
		h.append(adbt.fit(x_train, y_train, d[t]))
		for i in range(2098):
			if (adbt._predict(x_train[i]) != y_train[i]):
				e[t] = e[t] + d[t][i]
		# print(e[t])
		a[t] = 1/2*np.log( (( 1 - e[t]) / e[t]) )
		# print(a[t])

		if (t < L-1):
			for i in range(2098):
				if(adbt._predict(x_train[i]) == y_train[i]):
					d[t+1][i] = d[t][i] * np.exp(-a[t])
				else:
					d[t+1][i] = d[t][i] * np.exp(a[t])
				d[t+1] = (d[t+1]/d[t+1].sum())
	preds_train = []
	for i in range(2098):
		preds_train = np.append(preds_train, 0)
		for t in range(L):
			if (x_train[i][h[t].feature] >= h[t].split):
				preds_train[i] += a[t]*h[t].right_tree
			else:
				preds_train[i] += a[t]*h[t].left_tree
		if(preds_train[i] > 0):
			preds_train[i] = -1
		else:
			preds_train[i] = 1
				
	preds_test = []
	for i in range(700):
		preds_test = np.append(preds_test, 0)
		for t in range(L):
			if (x_test[i][h[t].feature] >= h[t].split):
				preds_test[i] += a[t]*h[t].right_tree
			else:
				preds_test[i] += a[t]*h[t].left_tree
		if(preds_test[i] > 0):
			preds_test[i] = -1
		else:
			preds_test[i] = 1

	# print(preds_train)
	# print(preds_test)
	train_accuracy = (preds_train == y_train).sum()/len(y_train)
	test_accuracy = (preds_test == y_test).sum()/len(y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	print('F1 Train {}'.format(f1(y_train, preds_train)))
	print('F1 Test {}'.format(f1(y_test, preds_test)))
	return train_accuracy, test_accuracy, f1(y_train, preds_train), f1(y_train, preds_train)
	
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
	
	# Enabled choosing different versions of RandomForest
	if args.random_forest == 1:
		OG_random_forest_testing(x_train, y_train, x_test, y_test)
	if args.random_forest == 2:
		random_forest_testing(x_train, y_train, x_test, y_test)
	if args.random_forest == 3:
		ft_random_forest_testing(x_train, y_train, x_test, y_test)

	if args.ada_boost == 1:
		x_train, y_train, x_test, y_test = adaboost_data(args.root_dir)
		tst_acc = []
		train_acc = []
		l = []
		f1_tr = []
		f1_tst = []
		for i in range(20):
			trn_acc, test_acc, f1_train, f1_test = ada_boost_testing(x_train, y_train, x_test, y_test, i*10)
			l.append(i*10)
			train_acc.append(trn_acc)
			tst_acc.append(test_acc)
			f1_tr.append(f1_train)
			f1_tst.append(f1_test)
		plt.plot(l, tst_acc)
		plt.xlabel("Number of features (L)")
		plt.ylabel("Testing Accuracy")
		plt.savefig("adaboosttest.png")
		plt.clf()
		plt.plot(l, train_acc)
		plt.xlabel("Number of Features (L)")
		plt.ylabel("Training Accuracy")
		plot.savefig("adaboosttrain.png")
		plt.clf()
		plotplot(l,f1_tr)
		plt.xlabel("Number of Features (L)")
		plt.ylabel("Training f1")
		plot.savefig("adaboosttrainf1.png")
		plot.clf()
		plotplot(l,f1_tst)
		plt.xlabel("Number of Features (L)")
		plt.ylabel("Testing f1")
		plot.savefig("adaboosttrainf1.png")
		plot.clf()
	
print('Done')
