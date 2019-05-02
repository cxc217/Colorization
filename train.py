import numpy as np
import os
from PIL import Image
from colorsys import hsv_to_rgb

def relu(x):
    return max(0,x)

def view3():
    print 'testing'

'''
#create a dim x dim size image
dim = 3
colors = []
for hue in range(dim):
    for sat in range(dim):
        # Convert color from HSV to RGB
        rgb = hsv_to_rgb(hue/dim, sat/dim, 1)
        rgb = [int(0.5 + 255*u) for u in rgb]
        colors.extend(rgb)

# Convert list to bytes
colors = bytes(colors)
img = Image.frombytes('RGB', (dim, dim), colors)
img.show()
img.save('hues.png')
'''

folder = ''
#for file in os.listdir(folder):
for x in range(0, 3):
    #im = Image.open(folder + '/' + 'hues.png')
    im = Image.open('hues.png')
    width, height = im.size
    print 'size: ' + str((width, height))
    
    # splits the image into 3 images, one for each color
    red, green, blue = im.split()
    
    red = red.load()
    green = green.load()
    blue = blue.load()
    
    array = np.arange(9).reshape(3,3)
    # iterate over all 3x3 grids
    for x in range(0, width-2):
        for y in range(0, height-2):
            # generate 3x3 grid
            for i in range(0, 3):
                for j in range(0, 3):
                    array[i,j] = red[i,j]
    break
