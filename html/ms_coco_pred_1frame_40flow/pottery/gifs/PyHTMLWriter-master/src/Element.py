import random
import string
from PIL import Image
import urllib2 as urllib
import io

class Element:
    """ A data element of a row in a table """
    def __init__(self, htmlCode = ""):
        self.htmlCode = htmlCode
        self.isHeader = False
        self.drawBorderColor = ''

    def imgToHTML(self, img_path, width = 200, overlay_path=None):
        res = '<img src="' + img_path.strip().lstrip() + '" width="' + str(width) + 'px" '
        if self.drawBorderColor:
            res += 'style="border: 10px solid ' + self.drawBorderColor + '" '
        if overlay_path:
          res += 'onmouseover="this.src=\'' + overlay_path.strip().lstrip() + '\';"'
          res += 'onmouseout="this.src=\'' + img_path.strip().lstrip() + '\';"'
        res += '/>'
        return res

    def imgToBboxHTML(self, img_path, bboxes, col='green', wid=300, ht=300, imsize = None):
        idd = "img_" + ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))

        # compute the ratios
        if imsize:
            actW = imsize[0]
            actH = imsize[1]
        else:
            actW, actH = self.tryComputeImgDim(img_path)
        actW = float(actW)
        actH = float(actH)
        if actW > actH:
            ht = wid * (actH / actW)
        else:
            wid = ht * (actW / actH)
        ratioX = wid / actW
        ratioY = ht / actH

        for i in range(len(bboxes)):
            bboxes[i] = [bboxes[i][0] * ratioX, bboxes[i][1] * ratioY, bboxes[i][2] * ratioX, bboxes[i][3] * ratioY]
        colStr = ''
        if self.drawBorderColor:
            col = self.drawBorderColor
            colStr = 'border: 10px solid ' + col + ';'
        htmlCode = """
            <canvas id=""" + idd + """ style="border:1px solid #d3d3d3; """ + colStr + """
                background-image: url(""" + img_path + """);
                background-repeat: no-repeat;
                background-size: contain;"
                width=""" + str(wid) + """,
                height=""" + str(ht) + """>
           </canvas>
           <script>
                var c = document.getElementById(\"""" + idd + """\");
                var ctx = c.getContext("2d");
                ctx.lineWidth="2";
                ctx.strokeStyle=\"""" + col + """\";"""
        for i in range(len(bboxes)):
            htmlCode += """ctx.rect(""" + ",".join([str(i) for i in bboxes[i]]) + """);"""
        htmlCode += """ctx.stroke();
        </script>
        """
        return htmlCode

    def imgToPosesHTML_SVG(self, img_path, poses, scale):
      ## TODO: THIS CODE IS BROKEN, maybe fix in the future (SVG is probably lighter than drawing with JS)
      connected = [[1, 2], [2, 3], # right leg
                   [11, 12], [12, 13], # right arm
                   [8, 13], # thorax - right arm
                   [3, 7], # right hip    - pelvis
                   [14, 15], [15, 16], # left arm
                   [8, 14], # thorax - left arm
                   [4, 5], [5, 6], # left leg
                   [4, 7], # left hip    - pelvis
                   [7, 8], # pelvis      - thorax
                   [8, 9], # thorax      - upper neck
                   [9, 10]] # upper neck - head 
      colors = ['red', 'red', # ... % right leg
                'red', 'red', # ... % right arm
                'red', # ... % thorax - right arm
                'red', # ... % right hip - pelvis
                'green', 'green', # ... % left arm
                'green', # ... % thorax - left arm
                'green', 'green', # ... % left leg
                'green', # ... % left hip - pelvis
                'yellow', # ... % pelvis - thorax
                'yellow', # ... % thorax - upper neck
                'yellow'] # % upper neck head
      htmlCode = '<div style="position: relative;"><img src="' + img_path + '" style="width:' + str(int(scale * 100)) + '%" /><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="position:absolute; left:0px; top:0px;"> <!-- <image xlink:href="' + img_path + '"  width="' + str(scale * 100) + '%" height="' + str(scale * 100) + '%" x=0 y=0 /> -->'
      for pose in poses:
        for cid, con in enumerate(connected):
          htmlCode += '<line x1="' + str(scale*pose[con[0]-1][0]) + '" y1="' + str(scale*pose[con[0]-1][1]) + '" x2="' + str(scale*pose[con[1]-1][0]) + '" y2="' + str(scale*pose[con[1]-1][1]) + '" stroke="' + colors[cid] + '" stroke-width="1" />'
      htmlCode += '</svg></div>'
      return htmlCode

    def imgToPosesHTML(self, img_path, poses, wid=300, ht=300, imsize = None, overlay_path = None):
      idd = "img_" + ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))

      # compute the ratios
      if imsize:
          actW = imsize[0]
          actH = imsize[1]
      else:
          actW, actH = self.tryComputeImgDim(img_path)
      actW = float(actW)
      actH = float(actH)
      if actW > actH:
          ht = wid * (actH / actW)
      else:
          wid = ht * (actW / actH)
      ratioX = wid / actW
      ratioY = ht / actH

      connected = [[1, 2], [2, 3], # right leg
                   [11, 12], [12, 13], # right arm
                   [8, 13], # thorax - right arm
                   [3, 7], # right hip    - pelvis
                   [14, 15], [15, 16], # left arm
                   [8, 14], # thorax - left arm
                   [4, 5], [5, 6], # left leg
                   [4, 7], # left hip    - pelvis
                   [7, 8], # pelvis      - thorax
                   [8, 9], # thorax      - upper neck
                   [9, 10]] # upper neck - head 
      colors = ['red', 'red', # ... % right leg
                'red', 'red', # ... % right arm
                'red', # ... % thorax - right arm
                'red', # ... % right hip - pelvis
                'green', 'green', # ... % left arm
                'green', # ... % thorax - left arm
                'green', 'green', # ... % left leg
                'green', # ... % left hip - pelvis
                'yellow', # ... % pelvis - thorax
                'yellow', # ... % thorax - upper neck
                'yellow'] # % upper neck head

      htmlCode = '<canvas id=' + idd + ' style="border:1px solid #d33d3; background-image: url(' + img_path + '); background-repeat: no-repeat; background-size: contain;" width="' + str(wid) + 'px" height="' + str(ht) + 'px"'
      if overlay_path:
        htmlCode += ' onMouseOver="this.style.backgroundImage=\'url(' + overlay_path + ')\'" onMouseOut="this.style.backgroundImage =\'url(' + img_path + ')\'"'
      htmlCode += ' ></canvas>' 
      htmlCode += """
           <script>
                var c = document.getElementById(\"""" + idd + """\");
                var ctx = c.getContext(\"2d\");
                ctx.lineWidth="2";"""
      for pose in poses:
        for cid,con in enumerate(connected):
          htmlCode += 'ctx.strokeStyle="' + colors[cid] + '"; ctx.beginPath(); ctx.moveTo(' + str(pose[con[0]-1][0]*ratioX) + ',' + str(pose[con[0]-1][1]*ratioY) + '); ctx.lineTo(' + str(pose[con[1]-1][0]*ratioX) + ',' + str(pose[con[1]-1][1]*ratioY) + '); ctx.stroke();'
      htmlCode += '</script>'
      return htmlCode

    def addImg(self, img_path, width = 200, bboxes=None, imsize=None, overlay_path=None, poses=None, scale=None):
        # bboxes must be a list of [x,y,w,h] (i.e. a list of lists)
        # imsize is the natural size of image at img_path.. used for putting bboxes, not required otherwise
        # even if it's not provided, I'll try to figure it out -- using the typical use cases of this software
        # overlay_path is image I want to show on mouseover
        if bboxes:
            # TODO overlay path not implemented yet for canvas image
            self.htmlCode += self.imgToBboxHTML(img_path, bboxes, 'green', width, width, imsize)
        elif poses:
            self.htmlCode += self.imgToPosesHTML(img_path, poses, width, width, imsize, overlay_path)
        else:
            self.htmlCode += self.imgToHTML(img_path, width, overlay_path)

    def addTxt(self, txt):
        if self.htmlCode: # not empty
                self.htmlCode += '<br />'
        self.htmlCode += str(txt)

    def getHTML(self):
        return self.htmlCode

    def setIsHeader(self):
        self.isHeader = True

    def setDrawCheck(self):
        self.drawBorderColor = 'green'
    
    def setDrawUnCheck(self):
        self.drawBorderColor = 'red'

    def setDrawBorderColor(self, color):
        self.drawBorderColor = color

    @staticmethod
    def getImSize(impath):
        im = Image.open(impath)
        return im.size

    @staticmethod
    def tryComputeImgDim(impath):
        try:
            im = Image.open(impath)
            res = im.size
            return res
        except:
            pass
        try:
            # most HACKY way to do this, remove the first '../'
            # since most cases
            impath2 = impath[3:]
            return self.getImSize(impath2)
        except:
            pass
        try:
            # read from internet
            fd = urllib.urlopen(impath)
            image_file = io.BytesIO(fd.read())
            im = Image.open(image_file)
            return im.size
        except:
            pass
        print 'COULDNT READ THE IMAGE SIZE!'

