import os
import numpy as np
import pylab as pl
from sklearn.svm import SVC
from math import sqrt, pow
import cPickle
DATA_DIR = '../data/normed_features/'
ANNOTATION_DIR = '../automated_annotation/'
f = open('svm_model')
feature_size = 14
svc = cPickle.load(f)
f.close()
data_dir = os.walk(DATA_DIR)
for root, dirs, files in os.walk(DATA_DIR):
	for file in files:
		inputs = open(root+file)
		x_feature = cPickle.load(inputs)
		svm_input = []
		for line in x_feature:
			svm_input.append(line[-14:])
		annotation = svc.predict(svm_input)
		f = open(ANNOTATION_DIR+file)
		cPickle.dump(annotation,f)
		f.close()