import os
import cPickle
import numpy as np
import random
X_DIR = "../data/normed_features/"
INPUT_DIR = "../data/normed_balanced_svm_input/"
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
		positive, negative, sampled_negative = [], [], []
		for data in x_feature:
			time = data[1]
			flag = False
			for interval in intervals:
				if interval[0] < time and time < interval[1]:
					flag = True
					break
			if flag:
				positive.append(np.append(data,[1]))
			else:
				negative.append(np.append(data,[0]))
		if len(positive) > len(negative):
			sampled_negative = negative
		else:
			sample_list = random.sample(range(0, len(negative)), len(positive))
			for ind in sample_list:
				sampled_negative.append(negative[ind])
		with open(INPUT_DIR+name,'w') as f:
			inputs = positive + sampled_negative
			random.shuffle(inputs)
			cPickle.dump(inputs,f)


