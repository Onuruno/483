import numpy as np
from PIL import Image
import time

def getHistograms(arr):
    histogram2 = np.empty([200,2], dtype=int)
    histogram4 = np.empty([200,4], dtype=int)
    histogram8 = np.empty([200,8], dtype=int)
    histogram16 = np.empty([200,16], dtype=int)
    for i in range(200):
        histogram2[i] = np.sum(np.bincount(arr[i], None, 256).reshape(-1, 128), axis=1)
        histogram4[i] = np.sum(np.bincount(arr[i], None, 256).reshape(-1, 64), axis=1)
        histogram8[i] = np.sum(np.bincount(arr[i], None, 256).reshape(-1, 32), axis=1)
        histogram16[i] = np.sum(np.bincount(arr[i], None, 256).reshape(-1, 16), axis=1)
    return histogram2, histogram4, histogram8, histogram16

def getHistogram(arr):
    histogram2 = np.sum(np.bincount(arr, None, 256).reshape(-1, 128), axis=1)
    histogram4 = np.sum(np.bincount(arr, None, 256).reshape(-1, 64), axis=1)
    histogram8 = np.sum(np.bincount(arr, None, 256).reshape(-1, 32), axis=1)
    histogram16 = np.sum(np.bincount(arr, None, 256).reshape(-1, 16), axis=1)
    return histogram2, histogram4, histogram8, histogram16

def getRgbValues(im):
    rgbr = []
    rgbg = []
    rgbb = []
    for i in range(96):
        for j in range(96):
            pixel = im.getpixel((i,j))
            rgbr.append(pixel[0])
            rgbg.append(pixel[1])
            rgbb.append(pixel[2])
    return rgbr, rgbg, rgbb

def compareImage(sourceR, sourceG, sourceB, imR, imG, imB):
    probRedS = sourceR/9216
    probRedQ = imR/9216
    probGreenS = sourceG/9216
    probGreenQ = imG/9216
    probBlueS = sourceB/9216
    probBlueQ = imB/9216
    klDivs = (np.sum(probRedQ*np.log(probRedQ/probRedS), axis=1) + np.sum(probGreenQ*np.log(probGreenQ/probGreenS), axis=1) + np.sum(probBlueQ*np.log(probBlueQ/probBlueS)))/3
    return np.argmin(klDivs)
    

t1 = time.time()
fileNames = []
with open("InstanceNames.txt") as file:
    fileNames = (file.read().splitlines())

rgbRed = []
rgbBlue = []
rgbGreen = []
for name in fileNames:
    rgbr = []
    rgbg = []
    rgbb = []
    im = Image.open("support_96/"+name)
    for i in range(96):
        for j in range(96):
            pixel=im.getpixel((i,j))
            rgbr.append(pixel[0])
            rgbg.append(pixel[1])
            rgbb.append(pixel[2])
    rgbRed.append(rgbr)
    rgbGreen.append(rgbg)
    rgbBlue.append(rgbb)
rgbRed = np.array(rgbRed)
rgbGreen = np.array(rgbGreen)
rgbBlue = np.array(rgbBlue)

Histogram128Red, Histogram64Red, Histogram32Red, Histogram16Red = getHistograms(rgbRed)
Histogram128Green, Histogram64Green, Histogram32Green, Histogram16Green = getHistograms(rgbGreen)
Histogram128Blue, Histogram64Blue, Histogram32Blue, Histogram16Blue = getHistograms(rgbBlue)

counter=0
#for i in range(200):
image1 = Image.open("query_1/"+fileNames[120])
imred, imgreen, imblue = getRgbValues(image1)
h128r, h64r, h32r, h16r = getHistogram(imred)
h128g, h64g, h32g, h16g = getHistogram(imgreen)
h128b, h64b, h32b, h16b = getHistogram(imblue)
    #if(i == compareImage(Histogram128Red, Histogram128Green, Histogram128Blue, h128r, h128g, h128b)):
        #counter+=1

t2=time.time()
print(t2-t1)
print(fileNames[120],fileNames[compareImage(Histogram128Red, Histogram128Green, Histogram128Blue, h128r, h128g, h128b)])
