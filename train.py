import numpy as np
from numpy import zeros
import os
import math
from PIL import Image
from colorsys import hsv_to_rgb

def relu(x):
    return max(0,x)

def color_cross_entropy(predicted_color, actual_color):
    pos = -1
    i = -1
    # only one position in vector will be a 1, all others will be 0
    for p in actual_color:
        i += 1
        if p == 1:
            pos = i
            break

    return -1 * math.log(predicted_color[pos])


# returns one hot label for color in palette
def one_hot(color, color_palette):
    i = -1
    for c in color_palette:
        i += 1
        if c == color:
            label = zeros(len(color_palette))
            label[i] = 1
            return label
    
    return 'error'



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
'''

color = 5
color_palette = [1,2,3,4,5,6,7]
label = one_hot(color, color_palette)

a = np.array([.02, .30, .45, .00, .25, .05, .00])#np.random.rand(len(color_palette))
print 'label: ' + str(label)
print 'a: ' + str(a)

print '\ncross-entropy: ' + str( color_cross_entropy(a, label))
