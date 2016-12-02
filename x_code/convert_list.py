import sys
import os
import cv2
import numpy as np


filename = '/nfs.yoda/xiaolonw/torch_projects/Frame_prediction_cGAN/trainlist_xy.txt'
filecenter = '/nfs.yoda/xiaolonw/torch_projects/Frame_prediction_cGAN/centers.txt'
fileout = '/nfs.yoda/xiaolonw/torch_projects/Frame_prediction_cGAN/trainlist_gap1_cluster.txt'



with open(filename, 'r') as f:
	filelist = f.readlines()
with open(filecenter, 'r') as f:
	centerlist = f.readlines()

listnum = len(filelist)

K = 40 
centers = np.zeros((K, 2)) 

for i in range(K):
	tc = centerlist[i]
	tc_set = tc.split()
	centers[i][0] = float(tc_set[0])
	centers[i][1] = float(tc_set[1])


counts = np.zeros(40)



with open(fileout, 'w') as f:

	for i in range(listnum):

		ts = filelist[i]
		ts_set = ts.split()

		filename1 = ts_set[0]
		filename2 = ts_set[1]
		xnum = float(ts_set[2])
		ynum = float(ts_set[3])

		current_sample = np.zeros((K, 2)) 
		current_sample[:, 0] = current_sample[:, 0] + xnum
		current_sample[:, 1] = current_sample[:, 1] + ynum

		dis = centers - current_sample

		dis = dis * dis
		dis = np.sum(dis, 1)
		cid = np.argmin(dis) + 1

		counts[cid - 1] = counts[cid - 1] + 1

		f.write('{0} {1} {2}\n'.format(filename1, filename2, cid) )


print(counts)



















