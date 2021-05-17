from mpi4py import MPI
import pandas as pd 
import numpy as np
import random
import matplotlib.pyplot as plt
import math
k = 3

data = pd.read_csv("Absenteeism_at_work.csv", sep=';') 
#commented df data below is used to create scatter plots to check if the centoids and clusters are updating
#df = pd.DataFrame(data[['Distance from Residence to Work','Age']])

df = pd.DataFrame(data)

mat = df.to_numpy()
data_size, attr_size = len(mat), len(mat[0])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
wt = MPI.Wtime()

def find_attr_mins_maxes(mat):
    attr_size = len(mat[0])
    mins, maxes = np.zeros(attr_size), np.zeros(attr_size)
    for i in range(attr_size):
        min_val, max_val = float('inf'), -float('inf')
        for j in range(data_size):
            min_val, max_val = min(min_val, mat[j][i]),max(max_val, mat[j][i])
        mins[i], maxes[i] = min_val, max_val
    return mins, maxes

def initialize_centoids(mins, maxes,k):
    centroids = np.zeros((k,attr_size))
    for i in range(k):
        for j in range(attr_size):
            centroids[i][j] = random.uniform(mins[j], maxes[j])
    return centroids

def cluster_data(data, centroids,k,start_indx):
    #initialize data for computation
    clusters = [None]*k
    for i in range(k): clusters[i] = []
    square_sums = [0]*k
    local_attr_sums = np.zeros((k,len(data[0])))
    
    for i in range(len(data)):
        min_dist, min_clust = float('inf'), float('inf')
        for x in range(k):
            summ = 0
            for j in range(len(data[0])):
                summ+=((data[i][j]-centroids[x][j])*(data[i][j]-centroids[x][j]))
            if summ<min_dist:
                min_dist,min_clust = summ, x
        z = i+start_indx
        clusters[min_clust].append(z)
        square_sums[min_clust]+=min_dist
        
        for j in range(len(data[0])):
            local_attr_sums[min_clust][j]+=data[i][j]
            
    return clusters, np.array(square_sums), local_attr_sums

def distribute_data(mat, k, clusters, mins, maxes, centroids):
    if rank == 0:
        data = split_data(mat,size)
        square_sums = np.zeros(k)
    else: 
        data= []
        square_sums = None
    
    #send data to other processes
    data = comm.scatter(data, root=0)
    centroids = comm.bcast(centroids, root=0)
    
    """
    calculate distances and create clusters
    based on the clusters, calculate their local sums to update centroids
    """
    clusters, sums, local_sums = cluster_data(data[0], centroids,k, data[1])
    #sent the calculated clusters, attribute sums, square sums to main process
    recv_clusters = comm.gather(clusters,0)  
    recv_new_centroids = comm.reduce(local_sums, op=MPI.SUM, root=0)
    comm.Reduce([sums, MPI.DOUBLE],[square_sums, MPI.DOUBLE],op = MPI.SUM,root = 0)
    
    combined_clusters = [None]*k
    for i in range(k): combined_clusters[i] = []
    
    if comm.rank==0:
        #combine cluster values received from each process
        for i in range(len(recv_clusters)):
            for x in range(len(recv_clusters[i])):
                combined_clusters[x]+=recv_clusters[i][x]
        
        #updates attribute sums for cluster by calculating their mean, and creating new centroids
        for x in range(len(recv_new_centroids)):
            l = len(combined_clusters[x])
            for j in range(len(recv_new_centroids[0])):
                if l>0:recv_new_centroids[x][j]/=l
                else:recv_new_centroids[x][j] = random.uniform(mins[j], maxes[j])

        print('Time taken to cluster data when running',size, 'process(es) parallel for cluster size',k,':', MPI.Wtime()-wt)
    return combined_clusters, recv_new_centroids, square_sums


def update_centroids(centroids, clusters,k, mins, maxes):
    for x in range(k):
        for j in range(attr_size):
            centroids[x][j] = 0
            for i in range(len(clusters[x])):
                centroids[x][j]+=mat[clusters[x][i]][j]
            centroids[x][j]/=(len(clusters[x]))
    return centroids


def gather_data(mat, clusters):
    gathered_data = []
    for x in range(len(clusters)):
        arr = []
        for i in range(len(clusters[x])):
            arr.append(list(mat[clusters[x][i]]))
        gathered_data.append(arr)
    return gathered_data

def plot_clusters(g, centroids):

    k = len(g)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Distance from Residence to Work')
    ax1.set_ylabel('Age')
    ax1.set_title('K means clustering for Distance from Residence to Work vs Age')

    
    colors = ["red", "green", "blue","yellow","darkblue","purple","cyan","orange","brown","pink"]
    for x in range(k):
        if(len(g[x])>0):
            ax1.scatter(np.array(g[x])[:,0], np.array(g[x])[:,1], s=10, c=colors[x], marker="o")
    ax1.scatter(np.array(centroids)[:,0], np.array(centroids)[:,1], s=10, c="black", marker="x",label="centroids")
    plt.legend(loc='upper right');
    plt.show()
        
        
def split_data(data, size):
    splitted = []
    step = math.ceil(len(data)/size)
    for i in range(size):
        start = i*step
        end = len(data) if (i+1)*step>len(data) else (i+1)*step
        splitted.append((data[start:end],start))
    return splitted

def converge(sum_quare_collection, square_sums,i):
    #print('Current sum squares', square_sums)
    converged_clusts = 0
    if i>0:
        differences = []
            #converged_clusts = 0
        for x in range(k):
            d = square_sums[x]-sum_quare_collection[-1][x]
            differences.append(d)
            if d>=0:
                converged_clusts+=1
        if converged_clusts==k:
            print('Program was converged after',i,'iterations')
            print('Sum of distances from prevoius iteration', sum_quare_collection[-1])
            print('Sum of distances from in this iteration', square_sums)
    sum_quare_collection.append(square_sums)
    return converged_clusts
        
mins, maxes = [],[]
centroids = []
clusters = []
g = 0

if rank == 0:
    mins, maxes = find_attr_mins_maxes(mat)
    centroids = initialize_centoids(mins, maxes,k)

sum_quare_collection = []

i = 0

lim = 20

while True:
    clusters, centroids_new, square_sums = distribute_data(mat, k, clusters, mins, maxes, centroids)
    #x = update_centroids(centroids, clusters,k, mins, maxes)
    converged_clusts = 0
    if rank==0:
        converged_clusts = converge(sum_quare_collection, square_sums, i)
    if rank==0 and len(df.columns)==2:
        g = gather_data(mat, clusters)
        plot_clusters(g,centroids)
        
    converged_clusts  = comm.bcast(converged_clusts, root=0)
    if converged_clusts ==k: break

    centroids = centroids_new
    i+=1
    
    


    
