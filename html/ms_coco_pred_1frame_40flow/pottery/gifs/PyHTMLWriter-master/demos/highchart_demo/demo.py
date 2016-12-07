import sys
sys.path.append('/home/rohytg/Software/PyHTMLWriter/src');
from Table import Table
from TableWriter import TableWriter

t = Table()
t.readFromCSV('table.csv', scale=100)
tw = TableWriter(t, 'out', makeChart = True)
tw.write()

