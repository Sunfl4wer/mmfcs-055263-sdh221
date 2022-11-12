from time import process_time_ns
from PIL import Image
from OneComponentAtATime import OneComponentAtATime as ocat
from TwoPass import TwoPass as tp
from ColumnRowSection import ColumnRowSection as crs
import numpy as np
import random
import matplotlib.pyplot as plt

def GetColors(labels):
    colors = []
    for i in labels:
        colors.append([random.randrange(1,255),
        random.randrange(1,255), random.randrange(1,255)])
    return colors

def GetRgbImage(labelToColor, labeled):
    shape = np.shape(labeled)
    rgb = np.zeros((shape[0], shape[1], 3), dtype=np.uint8) 
    for x in range(shape[0]):
        for y in range(shape[1]):
            label = labeled[x][y]
            if label == 0:
                c = [0,0,0]
            else:
                c = labelToColor[label]
            rgb[x][y][0] = c[0]
            rgb[x][y][1] = c[1]
            rgb[x][y][2] = c[2]
    return rgb

image = Image.open("pcb2.webp") # open colour image
gray = image.convert('L') # convert image to grayscale

# Let numpy do the heavy lifting for converting pixels to pure black or white
bw = np.asarray(gray).copy()

# Pixel range is 0...255, 256/2 = 128
bw[bw < 128] = 1  # Black
bw[bw >= 128] = 0 # White

ocatStart = process_time_ns()
labelsOCAT, labeledOCAT = ocat(bw)
ocatEnd = process_time_ns()

tpStart = process_time_ns()
labelsTP, labeledTP = crs(bw)
tpEnd = process_time_ns()

colorsOCAT = GetColors(labelsOCAT)
colorsTP = GetColors(labelsTP)
labelToColorOCAT = {labelsOCAT[i]: colorsOCAT[i] for i in range(len(colorsOCAT))}
labelToColorTP = {labelsTP[i]: colorsTP[i] for i in range(len(colorsTP))}
labeledImgOCAT = GetRgbImage(labelToColorOCAT, labeledOCAT)
labeledImgTP = GetRgbImage(labelToColorTP, labeledTP)

f, axarr = plt.subplots(2,2)
orgAx = axarr[0,0]
bwAx = axarr[0,1]
ocatAx = axarr[1,0]
tpAx = axarr[1,1]
orgAx.imshow(image)
bwAx.imshow(gray)
ocatAx.imshow(labeledImgOCAT)
tpAx.imshow(labeledImgTP)
orgAx.title.set_text('Original image')
ocatAx.title.set_text("One component at a time. Elapsed time: {0}s".format((ocatEnd-ocatStart)/10**9))
tpAx.title.set_text('Two pass. Elapsed time: {0}s'.format((tpEnd-tpStart)/10**9))
plt.show()

# Image.fromarray(coloredImg).save('pcb5-res.jpeg')
