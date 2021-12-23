import numpy as np
from PIL import Image

epsilon = 0.0001

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

def get3dGridHistograms(arr, gridSize, numberofBins):
    numberofgrids = (96//gridSize)*(96//gridSize)
    binInterval = 256//numberofBins
    histogram = np.empty([200,numberofgrids,numberofBins,numberofBins,numberofBins], dtype=int)
    for i in range(200):
        for j in range(numberofgrids):
            for k in range(gridSize*gridSize):
                pixel = arr[i][j][k]
                x = pixel[0]//binInterval
                y = pixel[1]//binInterval
                z = pixel[2]//binInterval
                histogram[i][j][x][y][z]+=1
    return histogram

def get3dGridHistogram(arr, gridSize, numberofBins):
    numberofgrids = (96//gridSize)*(96//gridSize)
    binInterval = 256//numberofBins
    histogram = np.zeros([numberofgrids,numberofBins,numberofBins,numberofBins], dtype=int)
    for j in range(numberofgrids):
        for k in range(gridSize*gridSize):
            pixel = arr[j][k]
            x = pixel[0]//binInterval
            y = pixel[1]//binInterval
            z = pixel[2]//binInterval
            histogram[j][x][y][z]+=1
    return histogram

def getGridHistograms(arr, gridSize, numberOfBins):
    numberofgrids = (96//gridSize)*(96//gridSize)
    binInterval = 256//numberOfBins
    histogram = np.empty([200,numberofgrids,numberOfBins], dtype=int)
    for i in range(200):
        for j in range(numberofgrids):
            histogram[i][j] = np.sum(np.bincount(arr[i][j], None, 256).reshape(-1, binInterval), axis=1)
    return histogram

def getGridHistogram(arr, gridSize, numberOfBins):
    numberofgrids = (96//gridSize)*(96//gridSize)
    binInterval = 256//numberOfBins
    histogram = np.empty([numberofgrids,numberOfBins], dtype=int)
    for j in range(numberofgrids):
        histogram[j] = np.sum(np.bincount(arr[j], None, 256).reshape(-1, binInterval), axis=1)
    return histogram

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

def getGridRgbValues(im, gridSize):
    rgbR = []
    rgbG = []
    rgbB = []
    rgb3D = []
    for i in range(len(im)):
        rgbr = []
        rgbg = []
        rgbb = []
        rgb3d = []
        for j in range(gridSize):
            for k in range(gridSize):
                pixel = im[i].getpixel((j,k))
                rgbr.append(pixel[0])
                rgbg.append(pixel[1])
                rgbb.append(pixel[2])
                rgb3d.append(pixel)
        rgbR.append(rgbr)
        rgbG.append(rgbg)
        rgbB.append(rgbb)
        rgb3D.append(rgb3d)
    return rgbR, rgbG, rgbB, rgb3D

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
    
    klDivs = (np.sum(probRedQ*np.log(probRedQ/probRedS), axis=1) + np.sum(probGreenQ*np.log(probGreenQ/probGreenS), axis=1) + np.sum(probBlueQ*np.log(probBlueQ/probBlueS), axis=1))/3
    return np.argmin(klDivs)

def compareGrid3dImage(source3d, im3d, gridSize):
    pixelPerGrid = gridSize*gridSize
    
    prob3dS = source3d/pixelPerGrid
    prob3dS[prob3dS==0] = epsilon
    
    prob3dQ = im3d/pixelPerGrid
    prob3dQ[prob3dQ==0] = epsilon
    
    klDivs = np.sum(np.sum(np.sum(np.sum(prob3dQ*np.log(prob3dQ/prob3dS), axis=2), axis=2), axis=2), axis=1)
    return np.argmin(klDivs)

def compareGridImage(sourceR, sourceG, sourceB, imR, imG, imB, gridSize):
    pixelPerGrid = gridSize*gridSize
    
    probRedS = sourceR/pixelPerGrid
    probRedS[probRedS==0] = epsilon
    
    probRedQ = imR/pixelPerGrid
    probRedQ[probRedQ==0] = epsilon
    
    probGreenS = sourceG/pixelPerGrid
    probGreenS[probGreenS==0] = epsilon
    
    probGreenQ = imG/pixelPerGrid
    probGreenQ[probGreenQ==0] = epsilon
    
    probBlueS = sourceB/pixelPerGrid
    probBlueS[probBlueS==0] = epsilon
    
    probBlueQ = imB/pixelPerGrid
    probBlueQ[probBlueQ==0] = epsilon
    
    klDivs = (np.sum(np.sum(probRedQ*np.log(probRedQ/probRedS), axis=2), axis=1) 
            + np.sum(np.sum(probGreenQ*np.log(probGreenQ/probGreenS), axis=2), axis=1) 
            + np.sum(np.sum(probBlueQ*np.log(probBlueQ/probBlueS), axis=2), axis=1))/3
    return np.argmin(klDivs)   

fileNames = []
with open("InstanceNames.txt") as file:
    fileNames = (file.read().splitlines())

####################################################### SINGLE GRID #######################################
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

print("Per channel accuricies without grids:")
print("Query_1 with 128, 64, 32, 16, 8 quantization respectively:")
print(q1accuricy128,q1accuricy64,q1accuricy32,q1accuricy16,q1accuricy8)
print("Query_2 with 128, 64, 32, 16, 8 quantization respectively:")
print(q2accuricy128,q2accuricy64,q2accuricy32,q2accuricy16,q2accuricy8)
print("Query_3 with 128, 64, 32, 16, 8 quantization respectively:")
print(q3accuricy128,q3accuricy64,q3accuricy32,q3accuricy16,q3accuricy8)

print("3D accuricies without grid:")
print("Query_1 with 128, 64, 32, 16 quantization respectively:")
print(q1_3daccuricy128,q1_3daccuricy64,q1_3daccuricy32,q1_3daccuricy16)
print("Query_2 with 128, 64, 32, 16 quantization respectively:")
print(q2_3daccuricy128,q2_3daccuricy64,q2_3daccuricy32,q2_3daccuricy16)
print("Query_3 with 128, 64, 32, 16 quantization respectively:")
print(q3_3daccuricy128,q3_3daccuricy64,q3_3daccuricy32,q3_3daccuricy16)


##################################### 48x48 GRID ####################################
rgbRed48 = []
rgbGreen48 = []
rgbBlue48 = []
rgb3d48 = []
for name in fileNames:
    rgb48r = []
    rgb48g = []
    rgb48b = []
    rgb48 = []
    imgrid = cropImage(Image.open("support_96/"+name), 48)
    for i in imgrid:
        subImageR = []
        subImageG = []
        subImageB = []
        subImage = []
        for j in range(48):
            for k in range(48):
                pixel = i.getpixel((j,k))
                subImageR.append(pixel[0])
                subImageG.append(pixel[1])
                subImageB.append(pixel[2])
                subImage.append(pixel)
        rgb48r.append(subImageR)
        rgb48g.append(subImageG)
        rgb48b.append(subImageB)
        rgb48.append(subImage)
    rgbRed48.append(rgb48r)
    rgbGreen48.append(rgb48g)
    rgbBlue48.append(rgb48b)
    rgb3d48.append(rgb48)
rgbRed48 = np.array(rgbRed48)
rgbGreen48 = np.array(rgbGreen48)
rgbBlue48 = np.array(rgbBlue48)
rgb3d48 = np.array(rgb3d48)

GridHistogram48Red = getGridHistograms(rgbRed48, 48, 8)
GridHistogram48Green = getGridHistograms(rgbGreen48, 48, 8)
GridHistogram48Blue = getGridHistograms(rgbBlue48, 48, 8)
GridHistogram3D48 = get3dGridHistograms(rgb3d48, 48, 16)

q1acc48=q2acc48=q3acc48=0
q13Dacc48=q23Dacc48=q33Dacc48=0

for i in range(200):
    image1 = cropImage(Image.open("query_1/"+fileNames[i]), 48)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 48)
    h48r = getGridHistogram(rgbR, 48, 8)
    h48g = getGridHistogram(rgbG, 48, 8)
    h48b = getGridHistogram(rgbB, 48, 8)
    h48 = get3dGridHistogram(rgb3D, 48, 16)
    if(i == compareGridImage(GridHistogram48Red, GridHistogram48Green, GridHistogram48Blue, h48r, h48g, h48b, 48)):
        q1acc48+=1
    if(i == compareGrid3dImage(GridHistogram3D48, h48, 48)):
        q13Dacc48+=1
        
    image1 = cropImage(Image.open("query_2/"+fileNames[i]), 48)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 48)
    h48r = getGridHistogram(rgbR, 48, 8)
    h48g = getGridHistogram(rgbG, 48, 8)
    h48b = getGridHistogram(rgbB, 48, 8)
    h48 = get3dGridHistogram(rgb3D, 48, 16)
    if(i == compareGridImage(GridHistogram48Red, GridHistogram48Green, GridHistogram48Blue, h48r, h48g, h48b, 48)):
        q2acc48+=1
    if(i == compareGrid3dImage(GridHistogram3D48, h48, 48)):
        q23Dacc48+=1
        
    image1 = cropImage(Image.open("query_3/"+fileNames[i]), 48)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 48)
    h48r = getGridHistogram(rgbR, 48, 8)
    h48g = getGridHistogram(rgbG, 48, 8)
    h48b = getGridHistogram(rgbB, 48, 8)
    h48 = get3dGridHistogram(rgb3D, 48, 16)
    if(i == compareGridImage(GridHistogram48Red, GridHistogram48Green, GridHistogram48Blue, h48r, h48g, h48b, 48)):
        q3acc48+=1
    if(i == compareGrid3dImage(GridHistogram3D48, h48, 48)):
        q33Dacc48+=1
q1acc48/=200
q2acc48/=200
q3acc48/=200
q13Dacc48/=200
q23Dacc48/=200
q33Dacc48/=200        

print("Per channel accuricies of query_1, query_2, query_3 with 48x48 grid:")
print(q1acc48, q2acc48, q3acc48)
print("3D accuricies of query_1, query_2, query_3 with 48x48 grid:")
print(q13Dacc48, q23Dacc48, q33Dacc48)        


##################################### 24x24 GRID ####################################
rgbRed24 = []
rgbGreen24 = []
rgbBlue24 = []
rgb3d24 = []
for name in fileNames:
    rgb24r = []
    rgb24g = []
    rgb24b = []
    rgb24 = []
    imgrid = cropImage(Image.open("support_96/"+name), 24)
    for i in imgrid:
        subImageR = []
        subImageG = []
        subImageB = []
        subImage = []
        for j in range(24):
            for k in range(24):
                pixel = i.getpixel((j,k))
                subImageR.append(pixel[0])
                subImageG.append(pixel[1])
                subImageB.append(pixel[2])
                subImage.append(pixel)
        rgb24r.append(subImageR)
        rgb24g.append(subImageG)
        rgb24b.append(subImageB)
        rgb24.append(subImage)
    rgbRed24.append(rgb24r)
    rgbGreen24.append(rgb24g)
    rgbBlue24.append(rgb24b)
    rgb3d24.append(rgb24)
rgbRed24 = np.array(rgbRed24)
rgbGreen24 = np.array(rgbGreen24)
rgbBlue24 = np.array(rgbBlue24)
rgb3d24 = np.array(rgb3d24)

GridHistogram24Red = getGridHistograms(rgbRed24, 24, 8)
GridHistogram24Green = getGridHistograms(rgbGreen24, 24, 8)
GridHistogram24Blue = getGridHistograms(rgbBlue24, 24, 8)
GridHistogram3D24 = get3dGridHistograms(rgb3d24, 24, 16)

q1acc24=q2acc24=q3acc24=0
q13Dacc24=q23Dacc24=q33Dacc24=0

for i in range(200):
    image1 = cropImage(Image.open("query_1/"+fileNames[i]), 24)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 24)
    h24r = getGridHistogram(rgbR, 24, 8)
    h24g = getGridHistogram(rgbG, 24, 8)
    h24b = getGridHistogram(rgbB, 24, 8)
    h24 = get3dGridHistogram(rgb3D, 24, 16)
    if(i == compareGridImage(GridHistogram24Red, GridHistogram24Green, GridHistogram24Blue, h24r, h24g, h24b, 24)):
        q1acc24+=1
    if(i == compareGrid3dImage(GridHistogram3D24, h24, 24)):
        q13Dacc24+=1
        
    image1 = cropImage(Image.open("query_2/"+fileNames[i]), 24)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 24)
    h24r = getGridHistogram(rgbR, 24, 8)
    h24g = getGridHistogram(rgbG, 24, 8)
    h24b = getGridHistogram(rgbB, 24, 8)
    h24 = get3dGridHistogram(rgb3D, 24, 16)
    if(i == compareGridImage(GridHistogram24Red, GridHistogram24Green, GridHistogram24Blue, h24r, h24g, h24b, 24)):
        q2acc24+=1
    if(i == compareGrid3dImage(GridHistogram3D24, h24, 24)):
        q23Dacc24+=1
        
    image1 = cropImage(Image.open("query_3/"+fileNames[i]), 24)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 24)
    h24r = getGridHistogram(rgbR, 24, 8)
    h24g = getGridHistogram(rgbG, 24, 8)
    h24b = getGridHistogram(rgbB, 24, 8)
    h24 = get3dGridHistogram(rgb3D, 24, 16)
    if(i == compareGridImage(GridHistogram24Red, GridHistogram24Green, GridHistogram24Blue, h24r, h24g, h24b, 24)):
        q3acc24+=1
    if(i == compareGrid3dImage(GridHistogram3D24, h24, 24)):
        q33Dacc24+=1
q1acc24/=200
q2acc24/=200
q3acc24/=200
q13Dacc24/=200
q23Dacc24/=200
q33Dacc24/=200        

print("Per channel accuricies of query_1, query_2, query_3 with 24x24 grid:")
print(q1acc24, q2acc24, q3acc24)
print("3D accuricies of query_1, query_2, query_3 with 24x24 grid:")
print(q13Dacc24, q23Dacc24, q33Dacc24)


##################################### 16x16 GRID ####################################
rgbRed16 = []
rgbGreen16 = []
rgbBlue16 = []
rgb3d16 = []
for name in fileNames:
    rgb16r = []
    rgb16g = []
    rgb16b = []
    rgb16 = []
    imgrid = cropImage(Image.open("support_96/"+name), 16)
    for i in imgrid:
        subImageR = []
        subImageG = []
        subImageB = []
        subImage = []
        for j in range(16):
            for k in range(16):
                pixel = i.getpixel((j,k))
                subImageR.append(pixel[0])
                subImageG.append(pixel[1])
                subImageB.append(pixel[2])
                subImage.append(pixel)
        rgb16r.append(subImageR)
        rgb16g.append(subImageG)
        rgb16b.append(subImageB)
        rgb16.append(subImage)
    rgbRed16.append(rgb16r)
    rgbGreen16.append(rgb16g)
    rgbBlue16.append(rgb16b)
    rgb3d16.append(rgb16)
rgbRed16 = np.array(rgbRed16)
rgbGreen16 = np.array(rgbGreen16)
rgbBlue16 = np.array(rgbBlue16)
rgb3d16 = np.array(rgb3d16)

GridHistogram16Red = getGridHistograms(rgbRed16, 16, 8)
GridHistogram16Green = getGridHistograms(rgbGreen16, 16, 8)
GridHistogram16Blue = getGridHistograms(rgbBlue16, 16, 8)
GridHistogram3D16 = get3dGridHistograms(rgb3d16, 16, 16)

q1acc16=q2acc16=q3acc16=0
q13Dacc16=q23Dacc16=q33Dacc16=0

for i in range(200):
    image1 = cropImage(Image.open("query_1/"+fileNames[i]), 16)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 16)
    h16r = getGridHistogram(rgbR, 16, 8)
    h16g = getGridHistogram(rgbG, 16, 8)
    h16b = getGridHistogram(rgbB, 16, 8)
    h16 = get3dGridHistogram(rgb3D, 16, 16)
    if(i == compareGridImage(GridHistogram16Red, GridHistogram16Green, GridHistogram16Blue, h16r, h16g, h16b, 16)):
        q1acc16+=1
    if(i == compareGrid3dImage(GridHistogram3D16, h16, 16)):
        q13Dacc16+=1
        
    image1 = cropImage(Image.open("query_2/"+fileNames[i]), 16)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 16)
    h16r = getGridHistogram(rgbR, 16, 8)
    h16g = getGridHistogram(rgbG, 16, 8)
    h16b = getGridHistogram(rgbB, 16, 8)
    h16 = get3dGridHistogram(rgb3D, 16, 16)
    if(i == compareGridImage(GridHistogram16Red, GridHistogram16Green, GridHistogram16Blue, h16r, h16g, h16b, 16)):
        q2acc16+=1
    if(i == compareGrid3dImage(GridHistogram3D16, h16, 16)):
        q23Dacc16+=1
        
    image1 = cropImage(Image.open("query_3/"+fileNames[i]), 16)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 16)
    h16r = getGridHistogram(rgbR, 16, 8)
    h16g = getGridHistogram(rgbG, 16, 8)
    h16b = getGridHistogram(rgbB, 16, 8)
    h16 = get3dGridHistogram(rgb3D, 16, 16)
    if(i == compareGridImage(GridHistogram16Red, GridHistogram16Green, GridHistogram16Blue, h16r, h16g, h16b, 16)):
        q3acc16+=1
    if(i == compareGrid3dImage(GridHistogram3D16, h16, 16)):
        q33Dacc16+=1
q1acc16/=200
q2acc16/=200
q3acc16/=200
q13Dacc16/=200
q23Dacc16/=200
q33Dacc16/=200        

print("Per channel accuricies of query_1, query_2, query_3 with 16x16 grid:")
print(q1acc16, q2acc16, q3acc16)
print("3D accuricies of query_1, query_2, query_3 with 16x16 grid:")
print(q13Dacc16, q23Dacc16, q33Dacc16)


##################################### 12x12 GRID ####################################
rgbRed12 = []
rgbGreen12 = []
rgbBlue12 = []
rgb3d12 = []
for name in fileNames:
    rgb12r = []
    rgb12g = []
    rgb12b = []
    rgb12 = []
    imgrid = cropImage(Image.open("support_96/"+name), 12)
    for i in imgrid:
        subImageR = []
        subImageG = []
        subImageB = []
        subImage = []
        for j in range(12):
            for k in range(12):
                pixel = i.getpixel((j,k))
                subImageR.append(pixel[0])
                subImageG.append(pixel[1])
                subImageB.append(pixel[2])
                subImage.append(pixel)
        rgb12r.append(subImageR)
        rgb12g.append(subImageG)
        rgb12b.append(subImageB)
        rgb12.append(subImage)
    rgbRed12.append(rgb12r)
    rgbGreen12.append(rgb12g)
    rgbBlue12.append(rgb12b)
    rgb3d12.append(rgb12)
rgbRed12 = np.array(rgbRed12)
rgbGreen12 = np.array(rgbGreen12)
rgbBlue12 = np.array(rgbBlue12)
rgb3d12 = np.array(rgb3d12)

GridHistogram12Red = getGridHistograms(rgbRed12, 12, 8)
GridHistogram12Green = getGridHistograms(rgbGreen12, 12, 8)
GridHistogram12Blue = getGridHistograms(rgbBlue12, 12, 8)
GridHistogram3D12 = get3dGridHistograms(rgb3d12, 12, 16)

q1acc12=q2acc12=q3acc12=0
q13Dacc12=q23Dacc12=q33Dacc12=0

for i in range(200):
    image1 = cropImage(Image.open("query_1/"+fileNames[i]), 12)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 12)
    h12r = getGridHistogram(rgbR, 12, 8)
    h12g = getGridHistogram(rgbG, 12, 8)
    h12b = getGridHistogram(rgbB, 12, 8)
    h12 = get3dGridHistogram(rgb3D, 12, 16)
    if(i == compareGridImage(GridHistogram12Red, GridHistogram12Green, GridHistogram12Blue, h12r, h12g, h12b, 12)):
        q1acc12+=1
    if(i == compareGrid3dImage(GridHistogram3D12, h12, 12)):
        q13Dacc12+=1
        
    image1 = cropImage(Image.open("query_2/"+fileNames[i]), 12)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 12)
    h12r = getGridHistogram(rgbR, 12, 8)
    h12g = getGridHistogram(rgbG, 12, 8)
    h12b = getGridHistogram(rgbB, 12, 8)
    h12 = get3dGridHistogram(rgb3D, 12, 16)
    if(i == compareGridImage(GridHistogram12Red, GridHistogram12Green, GridHistogram12Blue, h12r, h12g, h12b, 12)):
        q2acc12+=1
    if(i == compareGrid3dImage(GridHistogram3D12, h12, 12)):
        q23Dacc12+=1
        
    image1 = cropImage(Image.open("query_3/"+fileNames[i]), 12)
    rgbR, rgbG, rgbB, rgb3D = getGridRgbValues(image1, 12)
    h12r = getGridHistogram(rgbR, 12, 8)
    h12g = getGridHistogram(rgbG, 12, 8)
    h12b = getGridHistogram(rgbB, 12, 8)
    h12 = get3dGridHistogram(rgb3D, 12, 16)
    if(i == compareGridImage(GridHistogram12Red, GridHistogram12Green, GridHistogram12Blue, h12r, h12g, h12b, 12)):
        q3acc12+=1
    if(i == compareGrid3dImage(GridHistogram3D12, h12, 12)):
        q33Dacc12+=1
q1acc12/=200
q2acc12/=200
q3acc12/=200
q13Dacc12/=200
q23Dacc12/=200
q33Dacc12/=200        

print("Per channel accuricies of query_1, query_2, query_3 with 12x12 grid:")
print(q1acc12, q2acc12, q3acc12)
print("3D accuricies of query_1, query_2, query_3 with 12x12 grid:")
print(q13Dacc12, q23Dacc12, q33Dacc12)