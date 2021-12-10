import numpy as np
from PIL import Image
import time
epsilon = 0.0000001

def get3dHistograms(rgb3d):
    histogram2 = np.zeros([200,2,2,2], dtype=int)
    histogram4 = np.zeros([200,4,4,4], dtype=int)
    histogram8 = np.zeros([200,8,8,8], dtype=int)
    histogram16 = np.zeros([200,16,16,16], dtype=int)
    for i in range(200):
        for j in range(9216):
            x = rgb3d[i][j][0]//16
            y = rgb3d[i][j][1]//16
            z = rgb3d[i][j][2]//16
            histogram16[i][x][y][z]+=1
            histogram8[i][x//2][y//2][z//2]+=1
            histogram4[i][x//4][y//4][z//4]+=1
            histogram2[i][x//8][y//8][z//8]+=1
    return histogram2, histogram4, histogram8, histogram16

def get3dHistogram(rgb3d):
    histogram2 = np.zeros([2,2,2], dtype=int)
    histogram4 = np.zeros([4,4,4], dtype=int)
    histogram8 = np.zeros([8,8,8], dtype=int)
    histogram16 = np.zeros([16,16,16], dtype=int)
    for j in range(9216):
        x = rgb3d[j][0]//16
        y = rgb3d[j][1]//16
        z = rgb3d[j][2]//16
        histogram16[x][y][z]+=1
        histogram8[x//2][y//2][z//2]+=1
        histogram4[x//4][y//4][z//4]+=1
        histogram2[x//8][y//8][z//8]+=1
    return histogram2, histogram4, histogram8, histogram16
    
def getHistograms(arr):
    histogram2 = np.empty([200,2], dtype=int)
    histogram4 = np.empty([200,4], dtype=int)
    histogram8 = np.empty([200,8], dtype=int)
    histogram16 = np.empty([200,16], dtype=int)
    histogram32 = np.empty([200,32], dtype=int)
    for i in range(200):
        histogram2[i] = np.sum(np.bincount(arr[i], None, 256).reshape(-1, 128), axis=1)
        histogram4[i] = np.sum(np.bincount(arr[i], None, 256).reshape(-1, 64), axis=1)
        histogram8[i] = np.sum(np.bincount(arr[i], None, 256).reshape(-1, 32), axis=1)
        histogram16[i] = np.sum(np.bincount(arr[i], None, 256).reshape(-1, 16), axis=1)
        histogram32[i] = np.sum(np.bincount(arr[i], None, 256).reshape(-1, 8), axis=1)
    return histogram2, histogram4, histogram8, histogram16, histogram32

def getHistogram(arr):
    histogram2 = np.sum(np.bincount(arr, None, 256).reshape(-1, 128), axis=1)
    histogram4 = np.sum(np.bincount(arr, None, 256).reshape(-1, 64), axis=1)
    histogram8 = np.sum(np.bincount(arr, None, 256).reshape(-1, 32), axis=1)
    histogram16 = np.sum(np.bincount(arr, None, 256).reshape(-1, 16), axis=1)
    histogram32 = np.sum(np.bincount(arr, None, 256).reshape(-1, 8), axis=1)
    return histogram2, histogram4, histogram8, histogram16, histogram32

def cropImage(im, size):
    images=[]
    for i in range(96//size):
        for j in range(96//size):
            images.append(im.crop((i*size,j*size,(i+1)*size,(j+1)*size)))
    return images
        

def getRgbValues(im):
    rgbr = []
    rgbg = []
    rgbb = []
    rgb3d = []
    for i in range(96):
        for j in range(96):
            pixel = im.getpixel((i,j))
            rgbr.append(pixel[0])
            rgbg.append(pixel[1])
            rgbb.append(pixel[2])
            rgb3d.append(pixel)
    return rgbr, rgbg, rgbb, rgb3d

def compare3dImage(source3d, im3d):
    prob3dS = source3d/9216
    prob3dS[prob3dS==0] = epsilon
    
    prob3dQ = im3d/9216
    prob3dQ[prob3dQ==0] = epsilon
    
    klDivs = np.sum(np.sum(np.sum(prob3dQ*np.log(prob3dQ/prob3dS), axis=1), axis=1), axis=1)
    return np.argmin(klDivs)

def compareImage(sourceR, sourceG, sourceB, imR, imG, imB):
    probRedS = sourceR/9216
    probRedS[probRedS==0] = epsilon
    
    probRedQ = imR/9216
    probRedQ[probRedQ==0] = epsilon
    
    probGreenS = sourceG/9216
    probGreenS[probGreenS==0] = epsilon
    
    probGreenQ = imG/9216
    probGreenQ[probGreenQ==0] = epsilon
    
    probBlueS = sourceB/9216
    probBlueS[probBlueS==0] = epsilon
    
    probBlueQ = imB/9216
    probBlueQ[probBlueQ==0] = epsilon
    
    klDivs = (np.sum(probRedQ*np.log(probRedQ/probRedS), axis=1) + np.sum(probGreenQ*np.log(probGreenQ/probGreenS), axis=1) + np.sum(probBlueQ*np.log(probBlueQ/probBlueS)))/3
    return np.argmin(klDivs)
    

t1 = time.time()
fileNames = []
with open("InstanceNames.txt") as file:
    fileNames = (file.read().splitlines())

rgbRed = []
rgbBlue = []
rgbGreen = []
rgb3d = []
for name in fileNames:
    rgbr = []
    rgbg = []
    rgbb = []
    rgb = []
    im = Image.open("support_96/"+name)
    for i in range(96):
        for j in range(96):
            pixel=im.getpixel((i,j))
            rgb.append(pixel)
            rgbr.append(pixel[0])
            rgbg.append(pixel[1])
            rgbb.append(pixel[2])
    rgb3d.append(rgb)
    rgbRed.append(rgbr)
    rgbGreen.append(rgbg)
    rgbBlue.append(rgbb)
rgbRed = np.array(rgbRed)
rgbGreen = np.array(rgbGreen)
rgbBlue = np.array(rgbBlue)
rgb3d = np.array(rgb3d)

rgbRed48 = []
rgbGreen48 = []
rgbBlue48 = []
rgb3d48 = []
for name in fileNames:
    rgb48r = [[],[],[],[]]
    rgb48g = [[],[],[],[]]
    rgb48b = [[],[],[],[]]
    rgb48 = [[],[],[],[]]
    imgrid = cropImage(Image.open("support_96/"+name), 48)
    for i in range(4):
        for j in range(48):
            for k in range(48):
                pixel = imgrid[i].getpixel((j,k))
                rgb48[i].append(pixel)
                rgb48r[i].append(pixel[0])
                rgb48g[i].append(pixel[1])
                rgb48b[i].append(pixel[2])
    

for name in fileNames:
    rgbr = []
    rgbg = []
    rgbb = []
    rgb = []
    im = Image.open("support_96/"+name)
    for i in range(96):
        for j in range(96):
            pixel=im.getpixel((i,j))
            rgb.append(pixel)
            rgbr.append(pixel[0])
            rgbg.append(pixel[1])
            rgbb.append(pixel[2])
    rgb3d.append(rgb)
    rgbRed.append(rgbr)
    rgbGreen.append(rgbg)
    rgbBlue.append(rgbb)



'''
Histogram3d128, Histogram3d64, Histogram3d32, Histogram3d16 = get3dHistograms(rgb3d)

Histogram128Red, Histogram64Red, Histogram32Red, Histogram16Red, Histogram8Red = getHistograms(rgbRed)
Histogram128Green, Histogram64Green, Histogram32Green, Histogram16Green, Histogram8Green = getHistograms(rgbGreen)
Histogram128Blue, Histogram64Blue, Histogram32Blue, Histogram16Blue, Histogram8Blue = getHistograms(rgbBlue)

q1accuricy128=q1accuricy64=q1accuricy32=q1accuricy16=q1accuricy8=0
q2accuricy128=q2accuricy64=q2accuricy32=q2accuricy16=q2accuricy8=0
q3accuricy128=q3accuricy64=q3accuricy32=q3accuricy16=q3accuricy8=0

q1_3daccuricy128=q1_3daccuricy64=q1_3daccuricy32=q1_3daccuricy16=0
q2_3daccuricy128=q2_3daccuricy64=q2_3daccuricy32=q2_3daccuricy16=0
q3_3daccuricy128=q3_3daccuricy64=q3_3daccuricy32=q3_3daccuricy16=0

for i in range(200):
    image1 = Image.open("query_1/"+fileNames[i])
    imred, imgreen, imblue, im3d = getRgbValues(image1)
    h128r, h64r, h32r, h16r, h8r = getHistogram(imred)
    h128g, h64g, h32g, h16g, h8g = getHistogram(imgreen)
    h128b, h64b, h32b, h16b, h8b = getHistogram(imblue)
    h3d128, h3d64, h3d32, h3d16 = get3dHistogram(im3d)
    if(i == compareImage(Histogram128Red, Histogram128Green, Histogram128Blue, h128r, h128g, h128b)):
        q1accuricy128+=1
    if(i == compareImage(Histogram64Red, Histogram64Green, Histogram64Blue, h64r, h64g, h64b)):
        q1accuricy64+=1
    if(i == compareImage(Histogram32Red, Histogram32Green, Histogram32Blue, h32r, h32g, h32b)):
        q1accuricy32+=1
    if(i == compareImage(Histogram16Red, Histogram16Green, Histogram16Blue, h16r, h16g, h16b)):
        q1accuricy16+=1
    if(i == compareImage(Histogram8Red, Histogram8Green, Histogram8Blue, h8r, h8g, h8b)):
        q1accuricy8+=1
    if(i == compare3dImage(Histogram3d128, h3d128)):
        q1_3daccuricy128+=1
    if(i == compare3dImage(Histogram3d64, h3d64)):
        q1_3daccuricy64+=1
    if(i == compare3dImage(Histogram3d32, h3d32)):
        q1_3daccuricy32+=1
    if(i == compare3dImage(Histogram3d16, h3d16)):
        q1_3daccuricy16+=1
        
    image1 = Image.open("query_2/"+fileNames[i])
    imred, imgreen, imblue, im3d = getRgbValues(image1)
    h128r, h64r, h32r, h16r, h8r = getHistogram(imred)
    h128g, h64g, h32g, h16g, h8g = getHistogram(imgreen)
    h128b, h64b, h32b, h16b, h8b = getHistogram(imblue)
    h3d128, h3d64, h3d32, h3d16 = get3dHistogram(im3d)
    if(i == compareImage(Histogram128Red, Histogram128Green, Histogram128Blue, h128r, h128g, h128b)):
        q2accuricy128+=1
    if(i == compareImage(Histogram64Red, Histogram64Green, Histogram64Blue, h64r, h64g, h64b)):
        q2accuricy64+=1
    if(i == compareImage(Histogram32Red, Histogram32Green, Histogram32Blue, h32r, h32g, h32b)):
        q2accuricy32+=1
    if(i == compareImage(Histogram16Red, Histogram16Green, Histogram16Blue, h16r, h16g, h16b)):
        q2accuricy16+=1
    if(i == compareImage(Histogram8Red, Histogram8Green, Histogram8Blue, h8r, h8g, h8b)):
        q2accuricy8+=1
    if(i == compare3dImage(Histogram3d128, h3d128)):
        q2_3daccuricy128+=1
    if(i == compare3dImage(Histogram3d64, h3d64)):
        q2_3daccuricy64+=1
    if(i == compare3dImage(Histogram3d32, h3d32)):
        q2_3daccuricy32+=1
    if(i == compare3dImage(Histogram3d16, h3d16)):
        q2_3daccuricy16+=1
        
    image1 = Image.open("query_3/"+fileNames[i])
    imred, imgreen, imblue, im3d = getRgbValues(image1)
    h128r, h64r, h32r, h16r, h8r = getHistogram(imred)
    h128g, h64g, h32g, h16g, h8g = getHistogram(imgreen)
    h128b, h64b, h32b, h16b, h8b = getHistogram(imblue)
    h3d128, h3d64, h3d32, h3d16 = get3dHistogram(im3d)
    if(i == compareImage(Histogram128Red, Histogram128Green, Histogram128Blue, h128r, h128g, h128b)):
        q3accuricy128+=1
    if(i == compareImage(Histogram64Red, Histogram64Green, Histogram64Blue, h64r, h64g, h64b)):
        q3accuricy64+=1
    if(i == compareImage(Histogram32Red, Histogram32Green, Histogram32Blue, h32r, h32g, h32b)):
        q3accuricy32+=1
    if(i == compareImage(Histogram16Red, Histogram16Green, Histogram16Blue, h16r, h16g, h16b)):
        q3accuricy16+=1
    if(i == compareImage(Histogram8Red, Histogram8Green, Histogram8Blue, h8r, h8g, h8b)):
        q3accuricy8+=1
    if(i == compare3dImage(Histogram3d128, h3d128)):
        q3_3daccuricy128+=1
    if(i == compare3dImage(Histogram3d64, h3d64)):
        q3_3daccuricy64+=1
    if(i == compare3dImage(Histogram3d32, h3d32)):
        q3_3daccuricy32+=1
    if(i == compare3dImage(Histogram3d16, h3d16)):
        q3_3daccuricy16+=1
        
q1accuricy128/=200
q1accuricy64/=200
q1accuricy32/=200
q1accuricy16/=200
q1accuricy8/=200

q2accuricy128/=200
q2accuricy64/=200
q2accuricy32/=200
q2accuricy16/=200
q2accuricy8/=200

q3accuricy128/=200
q3accuricy64/=200
q3accuricy32/=200
q3accuricy16/=200
q3accuricy8/=200

q1_3daccuricy128/=200
q1_3daccuricy64/=200
q1_3daccuricy32/=200
q1_3daccuricy16/=200

q2_3daccuricy128/=200
q2_3daccuricy64/=200
q2_3daccuricy32/=200
q2_3daccuricy16/=200

q3_3daccuricy128/=200
q3_3daccuricy64/=200
q3_3daccuricy32/=200
q3_3daccuricy16/=200


print(q1accuricy128,q1accuricy64,q1accuricy32,q1accuricy16,q1accuricy8)
print(q2accuricy128,q2accuricy64,q2accuricy32,q2accuricy16,q2accuricy8)
print(q3accuricy128,q3accuricy64,q3accuricy32,q3accuricy16,q3accuricy8)

print(q1_3daccuricy128,q1_3daccuricy64,q1_3daccuricy32,q1_3daccuricy16)
print(q2_3daccuricy128,q2_3daccuricy64,q2_3daccuricy32,q2_3daccuricy16)
print(q3_3daccuricy128,q3_3daccuricy64,q3_3daccuricy32,q3_3daccuricy16)'''


t2=time.time()
print(t2-t1)
