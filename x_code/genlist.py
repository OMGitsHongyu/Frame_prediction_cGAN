import sys

filename = sys.argv[1]
rgbfolder = 'UCF101_frames_org2/ApplyEyeMakeup/'
flowfolder = 'UCF101_opt_flows_org2/ApplyEyeMakeup/'

with open(filename, 'r') as f:
	filelist = f.readlines()

listnum = len(filelist)

f = open(filename[:-4] + '_flow.txt', 'w')

for i in range(listnum):
	ts = filelist[i]
	ts_set = ts.split()
	frame_name1 = ts_set[0] 
	frame_name2 = ts_set[1]
	flownamex = frame_name1[:-4] + '_x.jpg' 
	flownamey = frame_name1[:-4] + '_y.jpg' 

	frame_name1 = rgbfolder + frame_name1
	frame_name2 = rgbfolder + frame_name2
	flownamex = flowfolder + flownamex
	flownamey = flowfolder + flownamey
	
	f.write('{0} {1} {2} {3}\n'.format(frame_name1, frame_name2, flownamex, flownamey) )

f.close()




