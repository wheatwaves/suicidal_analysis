import os
import numpy as np
import matplotlib.pyplot as plt
SU = ['CC0001', 'CC0005', 'CC0006', 'CC0008', 'CC0011', 'CC0013']
CO = ['CC0004', 'CC0020', 'CC0021', 'CC0022', 'CC0050', 'CC0051']
MH = ['CC0002', 'CC0010', 'CC0028', 'CC0031', 'CC0035', 'CC0036']
DATA_DIR = "../annotation/"
annotation_dir = os.walk(DATA_DIR)
su, co, mh = [], [], []
for root, dirs, files in annotation_dir:
	for file in files:
		name = file.split('.')[0]
		f = open(root + file)
# plot the box plot for time elapse
		# line_number = 0
		# for line in f.readlines():
		# 	line = line.strip().split('\t')
		# 	s, e = float(line[3]), float(line[5])
		# 	line_number += (e-s)
		line_number = len(f.readlines())
		if name in SU:
			su.append(line_number)
		if name in CO:
			co.append(line_number)
		if name in MH:
			mh.append(line_number)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([np.array(su),np.array(mh),np.array(co)])
plt.show()