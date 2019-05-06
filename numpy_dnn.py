import numpy as np
import operator
import math
import time
import os
from PIL import Image
from pre import img_to_arr
from pre import arr_to_img


def load():
    gray_image = 'gray_images'
    origin_image = 'images'
    pic = {}
    #get training and test data
    training_img = []
    test_img = []
    count = 0 #count 700 image for training img, the other is test.
    for file in os.listdir(gray_image):
        # loading image
        im = Image.open(gray_image + '/' + file)
        width, height = im.size
        # turn image to array
        pixels = img_to_arr(im)
        npPixels = np.asarray(pixels)
        npPixels = npPixels.reshape(height, width,3)
        if count <= 700:
            training_img.append(npPixels)
        else:
            test_img.append(npPixels)
        count += 1
    pic["training_images"] = np.asarray(training_img)
    pic["test_images"] = np.asarray(test_img)
    
    #get training and test labels
    training_lab = []
    test_lab = []
    count = 0 #count 700 image for training lab, the other is test.
    for file in os.listdir(origin_image):
        # loading image
        im = Image.open(origin_image + '/' + file)
        width, height = im.size
        # turn image to array
        pixels = img_to_arr(im)
        npPixels = np.asarray(pixels)
        npPixels = npPixels.reshape(height, width,3)
        if count <= 700:
            training_lab.append(npPixels)
        else:
            test_lab.append(npPixels)
        count += 1
    pic["training_labels"] = np.asarray(training_lab)
    pic["test_labels"] = np.asarray(test_lab)
    return pic["training_images"], pic["training_labels"], pic["test_images"], pic["test_labels"]

# Load Data
x_train, y_train, x_test, y_test = load()
x_train = (x_train/255.0).astype(float)
x_test = (x_test/255.0).astype(float)
y_train = (y_train/255.0).astype(float)
y_test = (y_test/255.0).astype(float)

def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = numpy.zeros((img.shape))
    #Looping through the image to apply the convolution operation.
    for r in numpy.uint16(numpy.arange(filter_size/2.0,
                                       img.shape[0]-filter_size/2.0+1)):
        for c in numpy.uint16(numpy.arange(filter_size/2.0,
                                           img.shape[1]-filter_size/2.0+1)):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on
            the image and filer sizes is the most tricky part of convolution.
            """
            curr_region = img[r-numpy.uint16(numpy.floor(filter_size/2.0)):r+numpy.uint16(numpy.ceil(filter_size/2.0)),
                                  c-numpy.uint16(numpy.floor(filter_size/2.0)):c+numpy.uint16(numpy.ceil(filter_size/2.0))]
            #Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = numpy.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
            #Clipping the outliers of the result matrix.
            final_result = result[numpy.uint16(filter_size/2.0):result.shape[0]-numpy.uint16(filter_size/2.0),
                                  numpy.uint16(filter_size/2.0):result.shape[1]-numpy.uint16(filter_size/2.0)]
    return final_result

# Define convolution layer
def c_forward(z, W, b):
    if len(z.shape) > 2 or len(W.shape) > 3: # Check if number of image channels matches the filter depth.
        if z.shape[-1] != W.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if W.shape[1] != W.shape[2]: # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if W.shape[1]%2==0: # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()
    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = numpy.zeros((z.shape[0]-W.shape[1]+1,
                            z.shape[1]-W.shape[1]+1,
                            W.shape[0]))

    # Convolving the image by the filter(s).
    for filter_num in range(W.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.
        """
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num],
                                    curr_filter[:, :, ch_num])
        else: # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter)
            feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.

def fc_backward(next_dz, W, z):
    N = z.shape[0]
    dz = np.dot(next_dz, W.T)  
    dw = np.dot(z.T, next_dz)  
    db = np.sum(next_dz, axis=0)  
    return dw / N, db / N, dz

# Define ReLU layer as activation 
def relu_forward(feature_map):
    #Preparing the output of the ReLU activation function.
    relu_out = numpy.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in numpy.arange(0,feature_map.shape[0]):
            for c in numpy.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])
    return relu_out
"""
def relu_backward(next_dz, z):
    dz = np.where(np.greater(z, 0), next_dz, 0)
    return dz


# Define cross_entropy as function 
def cross_entropy_loss(y_predict, y_true):
    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1,keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  
    dy = y_probability - y_true
    return loss, dy


# Define mean_square_error loss as function
def MSE_loss(y_predict, y_true):
    loss = np.mean(np.square(y_predict - y_true))
    dy = (2.0 * (y_predict - y_true))
    return loss, dy

# Initialize Weights and bias same as in pytorch 
# model size 784-200-50-10
weights = {}
input_units = 784
std1 = 1. / math.sqrt(200)
std2 = 1. / math.sqrt(50)
std3 = 1. / math.sqrt(10)
weights["W1"] = np.random.uniform(-std1, std1,(input_units, 200)).astype(np.float64)
weights["b1"] = np.random.uniform(-std1, std1, 200).astype(np.float64)
weights["W2"] = np.random.uniform(-std2, std2,(200, 50)).astype(np.float64)
weights["b2"] = np.random.uniform(-std2, std2, 50).astype(np.float64)
weights["W3"] = np.random.uniform(-std3, std3,(50, 10)).astype(np.float64)
weights["b3"] = np.random.uniform(-std3, std3, 10).astype(np.float64)


# Define forward and backward computation for dnn model
nuerons={}
gradients={}
def forward(X):
    nuerons["fc1"]=fc_forward(X.astype(np.float64),weights["W1"],weights["b1"])
    nuerons["fc1_relu"]=relu_forward(nuerons["fc1"])
    nuerons["fc2"]=fc_forward(nuerons["fc1_relu"],weights["W2"],weights["b2"])
    nuerons["fc2_relu"]=relu_forward(nuerons["fc2"])
    nuerons["y"]=fc_forward(nuerons["fc2_relu"],weights["W3"],weights["b3"])
    return nuerons["y"]

def backward(X,y_true):
    loss,dy=cross_entropy_loss(nuerons["y"],y_true) #use cross_entropy loss
    #loss, dy = MSE_loss(nuerons["y"],y_true) # use mean_square_eroor loss
    gradients["W3"],gradients["b3"],gradients["fc2_relu"]=fc_backward(dy,weights["W3"],nuerons["fc2_relu"])
    gradients["fc2"]=relu_backward(gradients["fc2_relu"],nuerons["fc2"])
    gradients["W2"],gradients["b2"],gradients["fc1_relu"]=fc_backward(gradients["fc2"],weights["W2"],nuerons["fc1_relu"])
    gradients["fc1"]=relu_backward(gradients["fc1_relu"],nuerons["fc1"])
    gradients["W1"],gradients["b1"],_=fc_backward(gradients["fc1"],weights["W1"],X)
    return loss

# Define SGD as optimizer
def _copy_weights_to_zeros(weights):
    result = {}
    result.keys()
    for key in weights.keys():
        result[key] = np.zeros_like(weights[key])
    return result
class SGD(object):
    def __init__(self, weights, lr=0.01):
        self.v = _copy_weights_to_zeros(weights)  
        self.iterations = 0 
        self.lr = lr

    def iterate(self, weights, gradients):
        for key in self.v.keys():
            self.v[key] = self.lr * gradients[key]
            weights[key] = weights[key] - self.v[key]
        self.iterations += 1


# Get prediction accuracy
def get_accuracy(X,y_true):
    y_predict=forward(X)
    return np.mean(np.equal(np.argmax(y_predict,axis=-1),
                            np.argmax(y_true,axis=-1)))

# Random get next batch
train_num = len(x_train)
def next_batch(batch_size):
    idx=np.random.choice(train_num,batch_size)
    return x_train[idx],y_train[idx]

# Set up hyperparameters
batch_size = 128
num_epoch = 10
sgd=SGD(weights,lr=0.1)

# Start Training
time0= time.time()
for e in range(num_epoch):
    for s in range(int(train_num/batch_size+1)):
        X,y=next_batch(batch_size)
        forward(X)
        loss=backward(X,y)
        sgd.iterate(weights,gradients)
    print("\n epoch:{} ; loss:{}".format(e+1,loss))
    print(" train_acc:{};  test_acc:{}".format(get_accuracy(X,y),get_accuracy(x_test,y_test)))

time1=time.time()
print("\n Final result test_acc:{}; ".format(get_accuracy(x_test,y_test)))
print ('Traning and Testing total excution time is: %s seconds ' % (time1-time0))   
"""
