import numpy as np
import math
import random
import os
from PIL import Image

MAX_ITERATIONS = 50

def find_clusters(data):
    iterations = 0
   
    # get a random number of clusters
    num_clusters = np.random.randint(low=30, high=50, size=1)
    print 'num_clusters: ' + str(num_clusters)
    
    # we store previous centers to see if any change in centers have been made
    previous_centers = []
    centers = pick_centers(data, num_clusters)
    
    
    while not (converged(previous_centers, centers, iterations)):
        iterations += 1
       
        print 'iteration: ' + str(iterations)
    
        clusters = [[] for i in range(num_clusters)]
        
        # assign data points to clusters
        clusters = find_closest_cluster(data, centers, clusters)
        
        # recalculate centers
        index = 0
        for cluster in clusters:
            previous_centers.append(centers[index])
            centers[index] = np.mean(cluster, axis=0).tolist()
            r,g,b = centers[index]
            r = int(r)
            g = int(g)
            b = int(b)
            centers[index] = (r,g,b)
            index += 1

    print("\nCenters: " + str(centers))
    
    # centers can be the k distinct colors 
    return centers


def distance(color1, color2):
    r = 2 * ((color1[0] - color2[0]) **2)
    g = 4 * ((color1[1] - color2[1]) **2)
    b = 3 * ((color1[2] - color2[2]) **2)

    return math.sqrt(r + g + b)


def pick_centers(data, num_clusters):
    arr = []
    # pick values from our data as centers
    for i in range(0, num_clusters):
        arr.append(data[random.randint(0, len(data)-1)])
    return arr

def converged(previous_centers, centers, iterations):
    if iterations >= MAX_ITERATIONS:
        return True
    return previous_centers == centers

def find_closest_cluster(data, centers, clusters):
    for data_point in data:
        # find the closest center for each data point
        cluster_index = min( [(i[0], np.linalg.norm(distance(data_point, centers[i[0]]))) for i in enumerate(centers)], key=lambda t:t[1])[0]
       
        # append data_point to cluster, if cluster is empty create list with data_point
        try:
            clusters[cluster_index].append(data_point)
        except KeyError:
            clusters[cluster_index] = [data_point]

    # Randomly add a datapoint to cluster if it is empty
    for cluster in clusters:
        if not cluster:
            cluster.append(data[random.randint(0, len(data)-1)])
    
    return clusters

# changes original picture colors to the ones found in clustering
def recolor(img, centers, dir, name):
    size = img.size
    mode = img.mode
    
    img_data = list(img.getdata())
    
    for i in range(0, len(img_data)):
        (r,g,b) = img_data[i]
        #replace rgb with values from centers
        shortest_dist = 999999999
        shortest_val = (0,0,0)
        for cent in centers:
            dist = distance((r,g,b), cent)
            if dist <= shortest_dist:
                shortest_dist = dist
                shortest_val = cent

        img_data[i] = shortest_val

    img2 = Image.new(mode, size)
    img2.putdata(img_data)
    
    img2_data = list(img2.getdata())

    img2.save(dir + name)

    return img2

# generate random indexes from folder
def Rand(folder, num):
    start = 0
    end = len(os.listdir(folder))
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return res


folder = 'images'
'''
# write centers of clusters to file so we dont have to do this everytime
filenames = os.listdir(folder)

# get 15 random pictures to use for clustering
positions = Rand(folder, 15)
pixels = []

for pos in positions:
    # loading image
    im = Image.open(folder + '/' + filenames[pos])
    pixels.extend(list(im.getdata()))

centers = find_clusters(pixels)

outF = open("centers.txt", "w")
for cent in centers:
    outF.write(str(cent))
    outF.write("\n")

outF.close()

'''
# read centers of clusters from fle
f1 = open('centers.txt', 'r')
lines = f1.readlines()

centers = []
for line in lines:
    line = line.replace('[', '').replace(']','').replace('\n','').replace('(','').replace(')','')
    line = line.split(',')
    r = int(line[0])
    g = int(line[1])
    b = int(line[2])
    centers.append((r,g,b))


recolor_dir = './recolor_images/'
if not os.path.exists(recolor_dir[0:-1]):
    os.makedirs(recolor_dir[0:-1])

for file in os.listdir(folder):
    im = Image.open(folder + '/' + file)
    recolor(im, centers, recolor_dir, file)
