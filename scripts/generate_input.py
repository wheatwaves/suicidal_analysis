import os
import cPickle
import numpy as np
X_DIR = "../data/normed_features/"
INPUT_DIR = "../data/svm_input/"
ANNOTATION_DIR = "../data/sen_annotation/"
annot_dir = os.walk(ANNOTATION_DIR)
for root, dirs, files in annot_dir:
	for file in files:
		name = file.split(".")[0]
		annotation = open(root+file)
		x_feature = []
		x_dir = os.walk(X_DIR)
		for a, b, c in x_dir:
			for d in c:
				if d.split(".")[0] == name:
					x = open(a+d)
					x_feature = cPickle.load(x)
		intervals = []
		for line in annotation.readlines():
			line = line.strip().split('\t')
			s, e = float(line[3]), float(line[5])
			intervals.append((s,e))
		svm_input = []
		for data in x_feature:
			time = data[1]
			flag = False
			for interval in intervals:
				if interval[0] < time and time < interval[1]:
					flag = True
					break
			if flag:
				svm_input.append(np.append(data,[1]))
			else:
				svm_input.append(np.append(data,[0]))
		with open(INPUT_DIR+name,'w') as f:
			cPickle.dump(svm_input,f)


