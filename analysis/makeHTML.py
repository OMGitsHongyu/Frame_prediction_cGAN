# This script is to make GIFs based on images and host them in HTML.
# Requirements:
# images2gif
# PyHTMLWriter: https://github.com/xiaolonw/PyHTMLWriter
# Note that you need to change Line 426:
# for im in images:
#    palettes.append( getheader(im)[1] )
# To
# for im in images:
#    palettes.append(im.palette.getdata()[1])
# according to http://stackoverflow.com/questions/19149643/error-in-images2gif-py-with-globalpalette

import sys
sys.path.append('../../src');
from Element import Element
from TableRow import TableRow
from Table import Table
from TableWriter import TableWriter
from images2gif import writeGif
from PIL import Image
import os

# write GIFs
for i in xrange(1,501):

    file_names = ['%04d' %i + '_input.jpg', '%04d' %i + '_pred.jpg']

    images = [Image.open(fn) for fn in file_names]

    assert images[0].size == images[1].size
    size = images[0].size

    for im in images:
        im.thumbnail(size, Image.ANTIALIAS)

    filename = '%04d' %i + '_input_pred.gif'
    writeGif(filename, images, duration=0.5)

# move GIFs to a dirctory
if not os.path.exists('./gifs/'):
    os.system('mkdir gifs')
os.system('mv *.gif gifs')

# Prepare files for HTML generator
with open('./gifs/PyHTMLWriter-master/demos/bbox_demo/gifs.txt', 'w') as f:
    for i in xrange(1,501):
        filename = '%04d' %i + '_input_pred.gif'
        f.write(filename+'\n')

# upload on a webpage
t = Table()
srcpath = './gifs/'
image_set_file = 'path_gifs.txt'
with open(image_set_file) as f:
    image_index = [x.strip() for x in f.readlines()]

trajnum = len(image_index) // 10

for r in range(trajnum):
    idx = r
#    if r == 0:
#        r = TableRow(isHeader = True)
#    else:
    r = TableRow()
    for e in range(10):
        idx2 = idx * 10 + e + 1
        e = Element()
        tpath = srcpath + '%04d' % idx2 + '_input_pred.gif'
        print tpath
        e.addImg(tpath)
        r.addElement(e)
    t.addRow(r)
tw = TableWriter(t, 'out')
tw.write()

