import sys
import os
import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten


filename = '/nfs.yoda/xiaolonw/torch_projects/Frame_prediction_cGAN/trainlist_xy.txt'
filecenter = '/nfs.yoda/xiaolonw/torch_projects/Frame_prediction_cGAN/centers.txt'
# fileout = '/nfs.yoda/xiaolonw/torch_projects/Frame_prediction_cGAN/trainlist_gap1_cluster.txt'



with open(filename, 'r') as f:
	filelist = f.readlines()

listnum = len(filelist)
K = 40
num_points = 160000
points = np.zeros((num_points, 2))


for i in range(num_points):
	ts = filelist[i]
	ts_set = ts.split()
	xnum = float(ts_set[2])
	ynum = float(ts_set[3])

	points[i][0] = xnum
	points[i][1] = ynum


init_centers = np.zeros((K, 2)) 
rp = np.random.permutation(num_points)
rp = rp[0 : K]

for i in range(K):
	pid = rp[i]
	init_centers[i] = np.copy(points[pid]) 

centers = kmeans(points, init_centers)
error   = centers[2]
centers = centers[1]
print(centers)
print(error)

with open(filecenter, 'w') as f:
	for i in range(K):
		f.write('{0} {1}\n'.format(centers[i][0], centers[i][1]) )

















