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
def arr_to_img(arr, img_name, width, height):
    mat = np.reshape(arr, (height, width))
    img = Image.fromarray(np.uint8(mat) , 'L')
    img.save(img_name + '_gray.jpg')
    #img.show()


# create new folders
pad_dir = './pad_images/'
if not os.path.exists(pad_dir[0:-1]):
	os.makedirs(pad_dir[0:-1])

gray_dir = './gray_images/'
if not os.path.exists(gray_dir[0:-1]):
    os.makedirs(gray_dir[0:-1])

count = 0
folder = 'images'
for file in os.listdir(folder):
	# loading image
	im = Image.open(folder + '/' + file)
	width, height = im.size

	# padding image
	new_im = Image.new("RGB", (width+2, height+2))
	new_im.paste(im, (1,1))
	new_im.save(pad_dir + file)

	# load new image with paddings
	im = new_im
	width, height = im.size

	# turn it into gray scale
	pixels = list(im.getdata())
	arr_to_img(rgbToGray(pixels), gray_dir+file[0:-4], width, height)

	count = count + 1
	if count % 50 == 0:
		print(str(count) +' images done')

