import os
import cPickle
import numpy as np
from math import sqrt
FEATURE_DIR = '../data/raw_features/'
INPUT_DIR = '../data/input_features/'
NORMED_DIR = '../data/normed_features/'
def format_row(line):
	s = str(line[0])
	for i in xrange(1, len(line)):
		s += ','
		s += str(line[i])
	return s
def feature_normalization(X, missing):
	m, n = len(X), len(X[0])
	A = np.zeros((m, n))
	for j in xrange(n):
		if j in missing:
			for i in xrange(m):
				A[i][j] = X[i][j]
			continue
		feature = []
		for i in xrange(m):
			feature.append(X[i][j])
		mean = np.mean(feature)
		var = np.var(feature)
		sigma = sqrt(var)
		for i in xrange(m):
			A[i][j] = (X[i][j] - mean) / sigma
	return A
feature_dir = os.walk(FEATURE_DIR)
for root, dirs, files in feature_dir:
	for file in files:
		name = file
		M = []
		print name
		with open(root + file) as f:
			lines = f.readlines()
			print len(lines)
			for line in lines[1:]:
				line = line.strip().split(',')
				for i in  xrange(len(line)):
					line[i] = float(line[i].strip())
				if line[1] > 120:
					break
				if line[3] == 1:
					M.append(line)
		if len(M) == 0:
			continue
		print len(M)
		A = feature_normalization(M, [0,1,2,3])
		# with open(INPUT_DIR + name, 'w') as g:
		# 	cPickle.dump(M, g)
		# with open(NORMED_DIR + name, 'w') as g:
		# 	cPickle.dump(A, g)