import sys
import os
import cv2
import numpy as np
import math

filename = sys.argv[1]
outname  = sys.argv[2] 
datasetpath = '/scratch/xiaolonw/videos/'
rgbfolder = 'UCF101_frames_org2/'
flowfolder = 'UCF101_opt_flows_org2/'

with open(filename, 'r') as f:
	filelist = f.readlines()

listnum = len(filelist)
jpggap = 10

f = open(outname, 'w')

for i in range(listnum):
	ts = filelist[i]
	ts_set = ts.split()
	video_name = ts_set[0]
	video_dir = datasetpath + rgbfolder  + video_name

	jpglist = os.listdir(video_dir)

	jpglen = len(jpglist) 
	samplenum = math.floor( ( jpglen - 15) / 10.0 )
	samplenum = int(samplenum)

	if i % 100 == 0:
		print(i)

	for j in range(samplenum): 
		jpgid = j * 10 
		jpgid2 = jpgid + jpggap

		frame_name1 = "%04d" % jpgid  + '.jpg'
		frame_name2 = "%04d" % jpgid2 + '.jpg'

		flownamex = frame_name1[:-4] + '_x.jpg' 
		flownamey = frame_name1[:-4] + '_y.jpg' 

		frame_name1 = rgbfolder + video_name + '/' + frame_name1
		frame_name2 = rgbfolder + video_name + '/' + frame_name2

		f.write('{0} {1} '.format(frame_name1, frame_name2) )


		flownamex = datasetpath + flowfolder + video_name + '/' + flownamex
		flownamey = datasetpath + flowfolder + video_name + '/' + flownamey

		im_flowx  = cv2.imread(flownamex)
		im_flowy  = cv2.imread(flownamey)

		flowx_num = np.mean(im_flowx) - 128
		flowy_num = np.mean(im_flowy) - 128

		f.write('{0} {1}\n'.format(flowx_num, flowy_num) )



f.close()




