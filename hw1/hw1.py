import numpy as np
from PIL import Image

fileNames = []
with open("InstanceNames.txt") as file:
    fileNames = (file.read().splitlines())

rgbValuesq1 = []
rgbValuesq2 = []   
rgbValuesq3 = []    
for name in fileNames:
    rgb1 = []
    rgb2 = []
    rgb3 = []
    im1 = Image.open("query_1/"+name)
    im2 = Image.open("query_2/"+name)
    im3 = Image.open("query_3/"+name)
    for i in range(96):
        for j in range(96):
            rgb1.append(im1.getpixel((i,j)))
            rgb2.append(im2.getpixel((i,j)))
            rgb3.append(im3.getpixel((i,j)))
    rgbValuesq1.append(rgb1)
    rgbValuesq2.append(rgb2)
    rgbValuesq3.append(rgb3)
rgbValuesq1 = np.array(rgbValuesq1)
rgbValuesq2 = np.array(rgbValuesq2)
rgbValuesq3 = np.array(rgbValuesq3)

print(len(rgbValuesq1))
print(len(rgbValuesq1[0]))
print(len(rgbValuesq2))
print(len(rgbValuesq2[0]))
print(len(rgbValuesq3))
print(len(rgbValuesq3[0]))