class TableRow:
    def __init__(self, isHeader = False, rno = -1):
        self.isHeader = isHeader
        self.elements = []
        self.rno = rno
    def addElement(self, element):
        self.elements.append(element)
    def getHTML(self):
        html = '<tr>'
        if self.rno >= 0:
          html += '<td><a href="#' + str(self.rno) + '">' + str(self.rno) + '</a>'
          html += '<a name=' + str(self.rno) + '></a></td>'
        for e in self.elements:
            if self.isHeader or e.isHeader:
                elTag = 'th'
            else:
                elTag = 'td'
            html += '<%s>' % elTag + e.getHTML() + '</%s>' % elTag
        html += '</tr>'
        return html

