import os
import sys
sys.path.append('../../src');
from Element import Element
from TableRow import TableRow
from Table import Table
from TableWriter import TableWriter

t = Table()
srcpath = 'snowboard/gifs/'
outpath = '../../../../../'
assert os.path.exists(outpath+srcpath)
image_set_file = 'gifs.txt'
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
tw = TableWriter(t, outpath)
tw.write()

