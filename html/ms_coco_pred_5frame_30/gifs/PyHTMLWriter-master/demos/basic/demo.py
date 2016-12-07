import sys
sys.path.append('/home/dragon123/Affordances/PyHTMLWriter/src');
from Element import Element
from TableRow import TableRow
from Table import Table
from TableWriter import TableWriter

t = Table()
for i in range(1):
    if i == 0:
        r = TableRow(isHeader = True)
    else:
        r = TableRow(rno = i)
    for e in range(10):
        e = Element()
        e.setDrawCheck()
        e.addImg('eiffeltower.jpg', bboxes=[[900,500,50,34],[100,100,100,100]])
        r.addElement(e)
    t.addRow(r)
r = TableRow(100)
for e in range(10):
    e = Element()
    e.setDrawCheck()
    e.addImg('eiffeltower.jpg', overlay_path='../taj.jpg')
    r.addElement(e)
t.addRow(r)

tw = TableWriter(t, 'out')
tw.write()

