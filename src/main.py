import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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

def random_forest_testing(x_train, y_train, x_test, y_test):
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
	return train_accuracy, test_accuracy
	

	

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
	if args.random_forest == 1:
		random_forest_testing(x_train, y_train, x_test, y_test)
	if args.ada_boost == 1:
		x_train, y_train, x_test, y_test = adaboost_data(args.root_dir)
		tst_acc = []
		l = []
		for i in range(20):
			trn_acc, test_acc = ada_boost_testing(x_train, y_train, x_test, y_test, i*10)
			l.append(i*10)
			tst_acc.append(test_acc)
		plt.plot(l, tst_acc)
		plt.xlabel("Number of features (L)")
		plt.ylabel("Testing Accuracy")
		plt.savefig("adaboosttest.png")
		plt.clf()
		

	print('Done')
	
	





