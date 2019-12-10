
import pandas as pd
import random
import statistics as stat
from scipy.spatial import distance_matrix

data = pd.read_csv("data.csv")

matrix = pd.DataFrame(distance_matrix(data.values, data.values), index=data.index, columns=data.index) # dissimilarity matrix

k = 2 # for k =2
DataPoint = []
DataPoint = random.sample(range(0,198),k) #generating random number 
cluster1,cluster2 = [],[]

def clustering(DataPoint):
    c1,c2 = [],[]
    for i in range(len(matrix.index)):
        temp1 = matrix.loc[i, DataPoint[0]]
        temp2 = matrix.loc[i, DataPoint[1]]
        if(temp1>temp2):
            c2.append(matrix.index[i])
        else:
            c1.append(matrix.index[i])
            
    findingCentroids(matrix,c1,0)
    findingCentroids(matrix,c2,1)
    return c1,c2

def findingCentroids(matrix,c1,x):  
    totalDistance = []
    for i in range(len(c1)): # calculate distance between cluster points for clusters
        distance  = []
        for j in range(len(c1)):
            distance.append(matrix.loc[c1[i],DataPoint[x]]) # for c1 datapoint1
           # distance.append(abs(matrix.loc[c1[i],DataPoint[x]] - matrix.loc[c1[j],DataPoint[x]]))
            d = sum(distance)
        temp = (d,c1[i])
        totalDistance.append(temp)  
            
    totalDistance.sort()
            
    DataPoint[x] = totalDistance[0][1]
#    
#distance in cluster
def distanceInCluster(cluster,Point):
    distance = 0
    for i in range(len(cluster)):
        distance = distance + matrix.loc[cluster[i],Point]
        
    return distance

def Silhouette(a,b):
    return ((b-a)/max(a,b))

    
    
for y in range(100): #finding datapoint after 100 iterations
    #print(y)
    cluster1,cluster2 = clustering(DataPoint)


widthCluster1 = 0
widthCluster2 = 0
fitness = []   
for y in range(len(cluster1)):    
    # FOR CLUSTER1
    #finding distance inside cluster
    #print(y , "cluster1")
    distance = distanceInCluster(cluster1,cluster1[y])
    a = distance/len(cluster1)
    #print("a done")
    
    #finding distance between clusters
    distance = distanceInCluster(cluster2,cluster1[y])
    b = distance/len(cluster2)
    #print("b done")
    widthCluster1 = widthCluster1 + Silhouette(a,b)
    
fitness.append(widthCluster1/ len(cluster1))
  
for y in range(len(cluster2)):    
    # FOR CLUSTER2
    #finding distance inside cluster
    #print(y  , "cluster2")
    distance = distanceInCluster(cluster2,cluster2[y])
    a = distance/len(cluster2)
   # print("a done")
    
    #finding distance between clusters
    distance = distanceInCluster(cluster1,cluster2[y])
    b = distance/len(cluster1)
   # print("b done")
    widthCluster2 = widthCluster2 + Silhouette(a,b)
    
fitness.append(widthCluster2/ len(cluster2))
    
SilhouetteWidth= stat.mean(fitness) 

cluster_ids = []

for y in range(data.shape[0]):
    if(data.index[y] in cluster1):
        cluster_ids.append(1)
    else:
        cluster_ids.append(2)
        
data['cluster_ids'] = cluster_ids
data.to_csv(' clusters_2.csv') 

print("Silhouette Width for k = 2 clusters is ", SilhouetteWidth)


#....................

k= 3
def kclustering(DataPoint):
    c1,c2,c3 =[],[],[]
    for i in range(len(matrix.index)):
        temp1 = matrix.loc[i, DataPoint[0]]
        temp2 = matrix.loc[i, DataPoint[1]]
        temp3 = matrix.loc[i, DataPoint[2]]
        
        if(temp1>temp2):
            if(temp3>temp2): 
                c2.append(matrix.index[i])
        if(temp2>temp1):
            if(temp3>temp1): 
                c1.append(matrix.index[i])
        if(temp1>temp3):
            if(temp2>temp3):
                c3.append(matrix.index[i])
                
    findingCentroids(matrix,c1,0)
    findingCentroids(matrix,c2,1)
    findingCentroids(matrix,c3,2)
    return c1,c2,c3
DataPoint = random.sample(range(0,198),k)

for y in range(100): #finding datapoint after 100 iterations
    #print(y)
    cluster1,cluster2,cluster3 = kclustering(DataPoint)



widthCluster1 = 0
widthCluster2 = 0
widthCluster3 = 0
fitness = [] 
  
for y in range(len(cluster1)):    
    # FOR CLUSTER1
    #finding distance inside cluster
    #print(y , "cluster1")
    distance = distanceInCluster(cluster1,cluster1[y])
    a = distance/len(cluster1)
    #print("a done")
    
    #finding distance between clusters
    distance = distanceInCluster(cluster2,cluster1[y])
    b = distance/len(cluster2)
    
    distance = distanceInCluster(cluster3,cluster1[y])
    c = distance/len(cluster3)
    #finding min
    b = min(b,c)
    
    
    #print("b done")
    widthCluster1 = widthCluster1 + Silhouette(a,b)
    
fitness.append(widthCluster1/ len(cluster1))
  
for y in range(len(cluster2)):    
    # FOR CLUSTER2
    #finding distance inside cluster
    #print(y  , "cluster2")
    distance = distanceInCluster(cluster2,cluster2[y])
    a = distance/len(cluster2)
   # print("a done")
    
    #finding distance between clusters
    distance = distanceInCluster(cluster1,cluster2[y])
    b = distance/len(cluster1)
    
    distance = distanceInCluster(cluster3,cluster2[y])
    c = distance/len(cluster3)
    #finding min
    b = min(b,c)
    
    
   # print("b done")
    widthCluster2 = widthCluster2 + Silhouette(a,b)
    
fitness.append(widthCluster2/ len(cluster2))
    
for y in range(len(cluster3)):    
    # FOR CLUSTER3
    #finding distance inside cluster
    #print(y  , "cluster2")
    distance = distanceInCluster(cluster3,cluster3[y])
    a = distance/len(cluster3)
   # print("a done")
    
    #finding distance between clusters
    distance = distanceInCluster(cluster1,cluster3[y])
    b = distance/len(cluster1)
    
    distance = distanceInCluster(cluster2,cluster3[y])
    c = distance/len(cluster2)
    #finding min
    b = min(b,c)
   # print("b done")
    widthCluster3 = widthCluster3 + Silhouette(a,b)
    
    
fitness.append(widthCluster3/ len(cluster3))

    
SilhouetteWidth= stat.mean(fitness) 

cluster_ids = []

for y in range(data.shape[0]):
    if(data.index[y] in cluster1):
        cluster_ids.append(1)
    elif(data.index[y] in cluster2):
        cluster_ids.append(2)
    else:
        cluster_ids.append(3)
        
data['cluster_ids'] = cluster_ids
data.to_csv(' clusters_3.csv') 

print("Silhouette Width for k = 3 clusters is ", SilhouetteWidth)







