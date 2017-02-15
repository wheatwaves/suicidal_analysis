# temporal function need to change afterwards
import os
import cPickle
import numpy as np
from math import ceil
total_time = 120
frame_rate = 30
feature_size = 14
total_frame = total_time * frame_rate
DATA_DIR = '../data/svm_input/'
SAMPLE_DIR = '../data/svm_input_with_framerate_30'
for root, dirs, files in os.walk(DATA_DIR):
	for file in files:
		data = cPickle.load(open(DATA_DIR+file))
		new_data = []
		data_size = len(data)
		num_frame = int(ceil(data_size / total_frame))
		for i in xrange(total_frame):
			new_frame = np.average(data[i*num_frame: min((i+1)*num_frame,data_size)], axis = 0)
			new_frame[0] = i
			new_frame[-1] = round(new_frame[-1])
			print new_frame
			new_data.append(new_frame)
		cPickle.dump(new_data, open(SAMPLE_DIR+file,'w'))