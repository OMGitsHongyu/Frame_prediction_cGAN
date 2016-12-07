import sys
sys.path.append('/home/rohit/Software/vis/PyHTMLWriter/src');
from Element import Element
from TableRow import TableRow
from Table import Table
from TableWriter import TableWriter
import numpy as np

t = Table()
for r in range(100):
    if r == 0:
        r = TableRow(isHeader = True)
    else:
        r = TableRow()
    for e in range(10):
        e = Element()
        pose = np.array([[587, 569, 490, 535, 621, 630, 512, 490, 488, 483, 576, 479, 457, 522, 550, 571, 491],[561, 447, 411, 393, 430, 537, 402, 298, 275, 201, 394, 377, 305, 289, 348, 367, 376]]).transpose().tolist()
        e.addImg("http://sun.pc.cs.cmu.edu/~rohit/Public/Images/himym_chairs.jpg", poses=[pose], width=500, imsize=[1280,768])
        
        r.addElement(e)
    t.addRow(r)
tw = TableWriter(t, 'out')
tw.write()

