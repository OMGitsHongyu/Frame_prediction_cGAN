import sys
import os

filedir = '/home/xiaolonw/video_dataset/hmdbtestTrainMulti_7030_split1/'
outlist = '/nfs.yoda/xiaolonw/torch_projects/Frame_prediction_cGAN/hmdb_cls.txt'

imgdir = '/scratch/xiaolonw/videos/'
jpgdir = 'hmdb_frames_org2/'
flowdir = 'hmdb_opt_flows_org2/'

classlist = 'hmdb_class_list.txt' 

filelist = os.listdir(filedir)

fout = open(outlist, 'w')
listnum = len(filelist)

fclass = open(classlist, 'w') 

print(filelist) 


for i in range(listnum):
	txtfile = filelist[i]
	txtfile2 = filedir + txtfile

	txt_set = txtfile.split('_')
	classname = ''

	txt_len = len(txt_set)
	for j in range(txt_len - 2):
		classname = classname + txt_set[j] 
		if j < txt_len - 3: 
			classname = classname + '_'


	fclass.write('{0} {1}\n'.format(classname, i) )

	with open(txtfile2, 'r') as f:
		samplelist = f.readlines()

	samplenum = len(samplelist)
	for j in range(samplenum):
		ts = samplelist[j]
		ts_set = ts.split()
		filename = ts_set[0]
		label = ts_set[1]
		if len(ts_set) > 2:
			print(ts)
			assert(len(ts_set) == 2)

		label = float(label)

		if label == 0:
			continue
		if label == 2:
			continue

		jpg_folder = imgdir + jpgdir + classname + '/' + filename + '/' 
		jpglist = os.listdir(jpg_folder)
		jpgnum  = len(jpglist)

		for k in range(jpgnum):
			jpgname = jpglist[k]
			flowxname = jpgname[:-4]  + '_x.jpg' 
			flowyname = jpgname[:-4]  + '_y.jpg' 

			jpgname = jpgdir + classname + '/' + filename + '/' + jpgname
			flowxname = flowdir + classname + '/' + filename + '/' + flowxname
			flowyname = flowdir + classname + '/' + filename + '/' + flowyname 

			fout.write('{0} {1} {2} {3}\n'.format(jpgname, flowxname, flowyname, i) )











