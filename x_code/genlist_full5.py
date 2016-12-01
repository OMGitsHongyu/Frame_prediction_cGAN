import sys
import os

filename = sys.argv[1]
outname  = sys.argv[2] 
datasetpath = '/scratch/xiaolonw/videos/'
rgbfolder = 'UCF101_frames_org2/'
flowfolder = 'UCF101_opt_flows_org2/'

with open(filename, 'r') as f:
	filelist = f.readlines()

listnum = len(filelist)
jpggap = 5

f = open(outname, 'w')

for i in range(listnum):
	ts = filelist[i]
	ts_set = ts.split()
	video_name = ts_set[0]
	video_dir = datasetpath + rgbfolder  + video_name

	jpglist = os.listdir(video_dir)

	jpglen = len(jpglist) 
	samplenum = round( ( jpglen - 10) / 10.0 )
	samplenum = int(samplenum)

	for j in range(samplenum): 
		jpgid = j * 10 
		jpgid2 = jpgid + jpggap

		frame_name1 = "%04d" % jpgid  + '.jpg'
		frame_name2 = "%04d" % jpgid2 + '.jpg'

		flownamex = frame_name1[:-4] + '_x.jpg' 
		flownamey = frame_name1[:-4] + '_y.jpg' 

		frame_name1 = rgbfolder + video_name + '/' + frame_name1
		frame_name2 = rgbfolder + video_name + '/' + frame_name2
		flownamex = flowfolder + video_name + '/' + flownamex
		flownamey = flowfolder + video_name + '/' + flownamey

		f.write('{0} {1} {2} {3}\n'.format(frame_name1, frame_name2, flownamex, flownamey) )

f.close()




