from PIL import Image
import numpy as np
import os

# change RBG array to gray scale array
def rgbToGray(pixels):
	gray_arr = []
	for i in range(0, len(pixels)):
		gray_arr.append(0.21*pixels[i][0] + 0.72*pixels[i][1] + 0.07*pixels[i][2])
	return gray_arr
# transform from array back to image
def arr_to_img(arr, img_name):
    mat = np.reshape(arr, (height, width))
    img = Image.fromarray(np.uint8(mat) , 'L')
    img.save(img_name + '_gray.jpg')
    #img.show()

# create a new folder
new_dir = './gray_images/'
if not os.path.exists(new_dir[0:-1]):
    os.makedirs(new_dir[0:-1])

count = 0
folder = 'images'
for file in os.listdir(folder):
	im = Image.open(folder + '/' + file)
	width, height = im.size
	pixels = list(im.getdata())
	arr_to_img(rgbToGray(pixels), new_dir+file[0:-4])
	count = count + 1
	if count % 50 == 0:
		print(str(count) +' images done')

