import numpy as np
import math
import random
import os
from PIL import Image

MAX_ITERATIONS = 50

def find_clusters(data):
    iterations = 0
   
    num_clusters = np.random.randint(low=30, high=50, size=1)
    print 'num_clusters: ' + str(num_clusters)
    
    previous_centers = []
    centers = pick_centers(data, num_clusters)
    
    while not (converged(previous_centers, centers, iterations)):
        iterations += 1
        #if iterations % 50 == 0:
        print 'iterations: ' + str(iterations)
    
        clusters = [[] for i in range(num_clusters)]
        
        # assign data points to clusters
        clusters = find_closest_cluster(data, centers, clusters)
        
        # recalculate centers
        index = 0
        #print 'len prv_centers: ' + str(len(previous_centers))
        #print 'len centers: ' + str(len(centers))
        for cluster in clusters:
            #previous_centers[index] = centers[index]
            previous_centers.append(centers[index])
            centers[index] = np.mean(cluster, axis=0).tolist()
            index += 1


    print("\nThe total number of data instances is: " + str(len(data)))
    print("The total number of iterations necessary is: " + str(iterations))
    print("\nThe means of each cluster are: " + str(centers))
    '''
    print("\nThe clusters are as follows:")
    for cluster in clusters:
        print("\nCluster with a size of " + str(len(cluster)) + " starts here:")
        print(np.array(cluster).tolist())
        print("Cluster ends here.")
    '''
    # centers can be the k distinct colors 
    return centers


def distance(color1, color2):
    r = 2 * ((color1[0] - color2[0]) **2)
    g = 4 * ((color1[1] - color2[1]) **2)
    b = 3 * ((color1[2] - color2[2]) **2)

    return math.sqrt(r + g + b)


def pick_centers(data, num_clusters):
    arr = []
    for i in range(0, num_clusters):
        arr.append(data[random.randint(0, len(data)-1)])
    return arr

def converged(previous_centers, centers, iterations):
    if iterations > MAX_ITERATIONS:
        return True
    return previous_centers == centers

def find_closest_cluster(data, centers, clusters):
    for instance in data:
        # Find which centroid is the closest
        # to the given data point.
        mu_index = min([(i[0], np.linalg.norm(distance(instance, centers[i[0]]))) \
                        for i in enumerate(centers)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.
    for cluster in clusters:
        if not cluster:
            cluster.append(data[random.randint(0, len(data)-1)])
    
    return clusters

# generate 5 random indexes from folder
def Rand(folder, num):
    start = 0
    end = len(os.listdir(folder))
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return res


folder = 'images'
filenames = os.listdir(folder)

# get 3 random pictures to use for clustering
positions = Rand(folder, 3)
pixels = []

for pos in positions:
    # loading image
    im = Image.open(folder + '/' + filenames[pos])
    pixels.extend(list(im.getdata()))

find_clusters(pixels)

