import numpy as np
import math
import random
import os
from PIL import Image

MAX_ITERATIONS = 5

def calc_gray(r,g,b):
    return 0.21*r + 0.72*g + 0.07*b

def find_clusters(data, num_clusters):
    iterations = 0
   
    # get a random number of clusters
    #num_clusters = np.random.randint(low=30, high=50, size=1)
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
            r,g,b, gray = centers[index]
            r = int(r)
            g = int(g)
            b = int(b)
            gray = int(gray)
            centers[index] = (r,g,b, gray)
            index += 1

    print("\nCenters: " + str(centers))
    
    # centers can be the k distinct colors 
    return centers


def distance(color1, color2):
    r = 2 * ((color1[0] - color2[0]) **2)
    g = 4 * ((color1[1] - color2[1]) **2)
    b = 3 * ((color1[2] - color2[2]) **2)
    gray = ((color1[3] - color2[3]) **2)

    return math.sqrt(r + g + b + gray)


def pick_centers(data, num_clusters):
    arr = []
    # pick values from our data as centers
    for i in range(0, num_clusters):
        index = random.randint(0, len(data)-1)
        arr.append(data[index])
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
        shortest_val = (0,0,0,0)
        for cent in centers:
            dist = distance((r,g,b, gray), cent)
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
    end = len(os.listdir(folder)) -1
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return res


def calc_error(centers):
    og_folder = 'images'
    og_filenames = os.listdir(og_folder)
    
    recolor_folder = 'recolor_images'
    recolor_filenames = os.listdir(recolor_folder)
    
    total_error = 0
    
    num_pics = 7
    positions = Rand(folder, num_pics)
    for pos in positions:
        #print 'pos: ' + str(pos)
        # loading image
        og_im = Image.open(og_folder + '/' + og_filenames[pos])
        recolor_im = Image.open(recolor_folder + '/' + recolor_filenames[pos])

        og_pixels = list(og_im.getdata())
        recolor_pixels = list(recolor_im.getdata())
        
        for i in range(0, len(og_pixels)):
            (r1,g1,b1) = og_pixels[i]
            (r2,g2,b2) = recolor_pixels[i]
            total_error += distance( (r1,g1,b1), (r2,g2,b2))

    # get the avg error
    return (1.0 * total_error) / num_pics

folder = 'images'

# write centers of clusters to file so we dont have to do this everytime
filenames = os.listdir(folder)

# get 2 random pictures to use for clustering
positions = Rand(folder, 1)
pixels = []

for pos in positions:
    # loading image
    im = Image.open(folder + '/' + filenames[pos])
    pixels.extend(list(im.getdata()))

for i in range(len(pixels)):
    (r,g,b) = pixels[i]
    gray = calc_gray(r,g,b)
    pixels[i] = (r,g,b,gray)


centers = find_clusters(pixels, 9)
i = 0
for num_clusters in [200]:
    i += 1
    centers = find_clusters(pixels,  num_clusters)

    outF = open('centers' + str(i) + '.txt', "w")
    for cent in centers:
        outF.write(str(cent))
        outF.write("\n")

    print 'i: ' + str(i) + '\tnum_clusters: ' + str(num_clusters) + '\terror: ' + str(calc_error(centers))
    outF.close()


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

