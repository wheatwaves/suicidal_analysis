import os
import numpy as np
from sklearn.svm import SVC
from math import sqrt, pow
import cPickle
SU = ['CC0001', 'CC0005', 'CC0006', 'CC0008', 'CC0011', 'CC0013']
CO = ['CC0004', 'CC0020', 'CC0021', 'CC0022', 'CC0050', 'CC0051']
MH = ['CC0002', 'CC0010', 'CC0028', 'CC0031', 'CC0035', 'CC0036']
DATA_DIR = '../data/svm_input/'
# leave last 2 file in each category for test, for training use 4-fold evaluation
kernel_parameter = ['poly','rbf','linear','sigmoid','precomputed']
c_parameter = range(-8,4)
feature_parameter = [1,2,14]

def svm(train_data, evaluation_data, kernel_name, c, feature_size):
	train_X, train_Y, evaluation_X, evaluation_Y = [], [], [], []
	for file_name in train_data:
		data = cPickle.load(open(DATA_DIR+file_name))
		for line in data:
			train_X.append(line[-1-feature_size:-1])
			train_Y.append(line[-1])
	for file_name in evaluation_data:
		data = cPickle.load(open(DATA_DIR+file_name))
		for line in data:
			evaluation_X.append(line[-1-feature_size:-1])
			evaluation_Y.append(line[-1])		
	svc = SVC(kernel = kernel_name, C = pow(10, c), class_weight = 'balanced')
	svc.fit(train_X, train_Y)
	predict_Y = svc.predict(evaluation_X)
	TP, FP ,FN = .0, .0, .0
	for i in xrange(len(evaluation_Y)):
		if predict_Y[i] == 1:
			if evaluation_Y[i] == 1:
				TP += 1
			else:
				FP += 1
		else:
			if evaluation_Y[i] == 1:
				FN += 1
	try:
		precision = TP / (TP + FP)
	except:
		precision = .0
	try:
		recall = TP / (TP + FN)
	except:
		recall = .0
	try:
		F1 = 2*precision*recall / (precision + recall)
	except:
		F1 = .0
	accuracy = svc.score(evaluation_X, evaluation_Y)
	return F1, accuracy, precision, recall
def cross_validation(kernel, c, feature_size):
	F1_list, accuracy_list, precision_list, recall_list = [], [], [], []
	for ind in xrange(4):
		train_SU, train_CO, train_MH = SU[:4], CO[:4], MH[:4]
		train_data, evaluation_data = [], []
		for group in [train_SU, train_CO, train_MH]:
			evaluation_data.append(group[ind])
			group.remove(group[ind])
			for item in group:
				train_data.append(item)
		print 'ind = ' + str(ind)
		F1, accuracy, precision, recall = svm(train_data, evaluation_data, kernel, c, feature_size)
		print 'F1 = ' + str(F1)
		print 'accuracy = ' + str(accuracy)
		print 'precision = ' + str(precision)
		print 'recall = ' + str(recall)
		F1_list.append(F1)
		accuracy_list.append(accuracy)
		precision_list.append(precision)
		recall_list.append(recall)
	return np.mean(F1_list), np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list)


def grid_search():
	best_kernel, best_c, best_feature_size = '', 0, 0
	best_F1 = 0
	print 'start grid search'
	for kernel in kernel_parameter:
		for c in c_parameter:
			for feature_size in feature_parameter:
				print '--'*32
				print 'kernel = ' + kernel
				print 'c = ' + str(c)
				print 'feature_size = ' + str(feature_size)
				F1, accuracy, precision, recall = cross_validation(kernel, c, feature_size)
				print 'overall_F1 = ' + str(F1)
				print 'overall_accuracy = ' + str(accuracy)
				print 'overall_precision = ' + str(precision)
				print 'overall_recall = ' + str(recall)
				if F1 > best_F1:
					best_F1 = F1
					best_kernel, best_c, best_feature_size = kernel, c, feature_size
	return best_F1, best_kernel, best_c, best_feature_size


if __name__ == '__main__':
	best_F1, best_kernel, best_c, best_feature_size = grid_search()
	print 'best_F1 = ' + str(best_F1)
	print 'best_kernel = ' + best_kernel
	print 'best_c = ' + str(best_c)
	print 'best_feature_size = ' + str(best_feature_size)
	F1, accuracy, precision, recall = svm(SU[:4]+CO[:4]+MH[:4],SU[4:]+CO[4:]+MH[4:])
	print 'test_F1 = '+str(F1)
	print 'test_accuracy = ' + str(accuracy)
	print 'test_precision = ' + str(precision)
	print 'test_recall = ' + str(recall)











