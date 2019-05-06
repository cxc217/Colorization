from PIL import Image
import numpy as np
import os

#transfer from image to array
def img_to_arr(img):
    pixels = list(img.getdata())
    return pixels

# change RBG array to gray scale array
def rgbToGray(pixels):
    gray_arr = [[0,0,0] for x in range(len(pixels))]
    for i in range(0, len(pixels)):
        gray_arr[i][0] = 0.21*pixels[i][0] + 0.72*pixels[i][1] + 0.07*pixels[i][2]
        gray_arr[i][1] = gray_arr[i][0]
        gray_arr[i][2] = gray_arr[i][1]
    return gray_arr

# transform from array back to image
def arr_to_img(arr, img_name, width, height):
    mat = np.reshape(arr, (height, width, 3))
    img = Image.fromarray(np.uint8(mat) , 'RGB')
    img.save(img_name + '_colored.jpg')
    #img.show()



