from typing import Tuple, List
from math import sqrt
import random
import matplotlib.pyplot as plt




def readfile(filename: str) -> Tuple[List, List, List]:
    headers = None
    row_names = list()
    data = list()

    with open(filename) as file_:
        for line in file_:
            values = line.strip().split("\t")
            if headers is None:
                headers = values[1:]
            else:
                row_names.append(values[0])
                data.append([float(x) for x in values[1:]])
    return row_names, headers, data


# .........DISTANCES........
# They are normalized between 0 and 1, where 1 means two vectors are identical
def euclidean(v1, v2):
    distance = 0  # TODO
    return 1 / (1+distance)

def euclidean_squared(v1, v2):
    return euclidean(v1, v2)**2

def pearson(v1, v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)
    # Sums of squares
    sum1sq = sum([v**2 for v in v1])
    sum2sq = sum([v**2 for v in v2])
    # Sum of the products
    products = sum([a * b for (a, b) in zip(v1, v2)])
    # Calculate r (Pearson score)
    num = products - (sum1 * sum2 / len(v1))
    den = sqrt((sum1sq - sum1**2 / len(v1)) * (sum2sq - sum2**2 / len(v1)))
    if den == 0:
        return 0
    return 1 - num / den


# ........HIERARCHICAL........
class BiCluster:
    def __init__(self, vec, left=None, right=None, dist=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = dist
        
def hcluster(rows, distance=pearson):
    distances = {}  # Cache of distance calculations
    currentclustid = -1  # Non original clusters have negative id

    # Clusters are initially just the rows
    clust = [BiCluster(row, id=i) for (i, row) in enumerate(rows)]

    """
    while ...:  # Termination criterion
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                distances[(clust[i].id, clust[j].id)] = ...

            # update closest and lowestpair if needed
            ...
        # Calculate the average vector of the two clusters
        mergevec = ...

        # Create the new cluster
        new_cluster = BiCluster(...)

        # Update the clusters
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(new_cluster)
    """

    return clust[0]

def printclust(clust: BiCluster, labels=None, n=0):
    # indent to make a hierarchy layout
    indent = " " * n
    if clust.id < 0:
        # Negative means it is a branch
        print(f"{indent}-")
    else:
        # Positive id means that it is a point in the dataset
        if labels == None:
            print(f"{indent}{clust.id}")
        else:
            print(f"{indent}{labels[clust.id]}")
    # Print the right and left branches
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n+1)


# ......... K-MEANS ..........
def kcluster(rows, distance=pearson, k=4,try_times = 5):
    ranges=[(min([row[i] for row in rows]),
    max([row[i] for row in rows])) for i in range(len(rows[0]))]
    # Create k randomly placed centroids
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]
    lastmatches=None
    best_distances={}
    for ty in range(try_times):
        distances={}
        for j in range(len(rows)):
            row=rows[j]
            bestmatch=0
            for i in range(k):
                d=distance(clusters[i],row)
                d2 = distance(clusters[bestmatch],row)
                if d<d2: 
                    bestmatch=i
                    best_dist = d
                else:
                    best_dist = d2
            distances[j] = best_dist
        minus = 0
        count = 0
        if ty == 0:
            best_distances = distances
            best_clusters = clusters
        else:
            for n in range(len(rows)):
                count+=1
                if best_distances[n] > distances[n]:
                    minus+=1
            if count/2 < float(minus):
                best_distances = distances
                best_clusters = clusters
        clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]      

    clusters = best_clusters
    for t in range(100):
        bestmatches=[[] for i in range(k)]
    # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row=rows[j]
            bestmatch=0
            for i in range(k):
                d=distance(clusters[i],row)
                if d<distance(clusters[bestmatch],row): bestmatch=i
            bestmatches[bestmatch].append(j)
        # If the results are the same as last time, done
        if bestmatches==lastmatches: break
        lastmatches=bestmatches
        # Move the centroids to the average of their members
        for i in range(k):
            avgs=[0.0]*len(rows[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j]/=len(bestmatches[i])
                clusters[i]=avgs
    total = list()
    i = -1
    centr = list()
    centr_list = list()
    for bestm in bestmatches:
        dist = 0
        i+=1
        if len(bestm)!= 0:
            centr.append(clusters[i])
            for best in bestm:
                dist += pearson(clusters[i],rows[best]) #No entiendo como calcular la distancia con euclidean, por eso uso pearson
            centr_list.append(dist)
    total.append(centr)
    total.append(centr_list)

    total_return = tuple(total)
    
    return total_return


    

    #kclus = tuple()


def main():
    """
    blognames, word, data = readfile('blogdata.txt')
    kclust = kcluster(data,k=10)
    print([blognames[r] for r in kclust[0]])
    print([blognames[r] for r in kclust[1]])    
    """
    blognames, word, data = readfile('blogdata_full.txt')
    #centroides,distances = kcluster(data,k=10,try_times=5)
    distortions = []
    krange = range(1,10)
    print("Sum of distances for n 'k' ")
    for kmean in krange:
        _,distances = kcluster(data,k=kmean,try_times=5)
        distortions.append(sum(distances))
        print(distortions[kmean-1])
    plt.figure(figsize=(16,8))
    plt.plot(krange,distortions, 'bx-')
    plt.xlabel('krange')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    """
    for i in range(len(centroides)):
        print("For centroid: " + str(centroides[i]) + " have a total distance of: " + str(distances[i]))
    """
    
    
if __name__=="__main__":
    main()


