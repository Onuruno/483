import numpy as np
from PIL import Image

fileNames = []
with open("InstanceNames.txt") as file:
    fileNames = (file.read().splitlines())

rgbValues = []
for name in fileNames:
    rgb = []
    im = Image.open("support_96/"+name)
    for i in range(96):
        for j in range(96):
            rgb.append(im.getpixel((i,j)))
    rgbValues.append(rgb)
rgbValues = np.array(rgbValues)

Histogram128 = np.zeros(shape=(200,3,2), dtype=int)
Histogram64 = np.zeros(shape=(200,3,4), dtype=int)
Histogram32 = np.zeros(shape=(200,3,8), dtype=int)
Histogram16 = np.zeros(shape=(200,3,16), dtype=int)

for i in range(len(rgbValues)):
    for pixel in range(96*96):
        Histogram128[i][0][rgbValues[i][pixel][0]//128] +=1
        Histogram128[i][1][rgbValues[i][pixel][1]//128] +=1
        Histogram128[i][2][rgbValues[i][pixel][2]//128] +=1
        
        Histogram64[i][0][rgbValues[i][pixel][0]//64] +=1
        Histogram64[i][1][rgbValues[i][pixel][1]//64] +=1
        Histogram64[i][2][rgbValues[i][pixel][2]//64] +=1
        
        Histogram32[i][0][rgbValues[i][pixel][0]//32] +=1
        Histogram32[i][1][rgbValues[i][pixel][1]//32] +=1
        Histogram32[i][2][rgbValues[i][pixel][2]//32] +=1
        
        Histogram16[i][0][rgbValues[i][pixel][0]//16] +=1
        Histogram16[i][1][rgbValues[i][pixel][1]//16] +=1
        Histogram16[i][2][rgbValues[i][pixel][2]//16] +=1
        
