import numpy as np
import cv2
from mpi4py import MPI
import math
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import pandas as pd

# Load an color image in grayscale
#reading Image 
img = cv2.imread('spring.jpeg')
#plt.imshow(img, interpolation="none")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Counting frequency of color scales using MPI
def count_frequency(arr):
    wt = MPI.Wtime()
    res = Counter([])
    if rank == 0:
        data = np.array(arr).flatten()
    else:
        data = []
    data = comm.scatter(split_data(data,size), root=0)

    recvbuf = comm.gather(Counter(data),0)   
    if comm.rank==0:
        for d in recvbuf:
            res=res+d
    wt = MPI.Wtime() - wt
    print('Time taken to count frequency of color scales: ', wt)
    return res

def visualize_data(data):
    freq = count_frequency(data)
    df = pd.DataFrame.from_dict(freq, orient='index', columns={'frequency'}).reset_index()
    df = df.rename(columns={'index':'color scale', 0:'count'})
    #bar chart for different colo scales
    #df.plot.bar(title = 'Frequency of color scales',x='color scale', y='frequency', color='Red',rot=30)    
    
    #Histogram for color frequencies        
    pd.DataFrame(np.array(data).flatten()).plot.hist(title='Histogram of color scales',alpha=0.7);
 

def img_show(img):
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('messigray.png', img)
        cv2.destroyAllWindows()
        
def split_data(data, size):
    splitted = []
    step = math.ceil(len(data)/size)
    for i in range(size):
        start = i*step
        end = len(data) if (i+1)*step>len(data) else (i+1)*step
        #print('i', i, data[start:end])
        splitted.append(data[start:end])
    return splitted



def image_scale_grey(img):
    wt = MPI.Wtime()
    if rank == 0: data = np.asarray(img)
    else: data = []
    data = comm.scatter(split_data(data,size), root=0)

    r, c = len(data), len(data[0])
    v = np.empty((r,c))
    for i in range(r):
        for j in range(c):
            v[i][j] = int((0.299*data[i][j][2])+(0.587*data[i][j][0])+(0.114*data[i][j][1]))
    
    recvbuf = comm.gather(v,0)
    res = []
    if comm.rank==0:
        for i in range(len(recvbuf)):
            for r in recvbuf[i]:
                res.append(r)  
        imagegray = np.asarray(res)
        cv2.imwrite('springgray.png', imagegray)
        plt.imshow(imagegray,cmap='gray')
    wt = MPI.Wtime() - wt
    print('Time taken to scale the image to grey: ', wt)
    return res

wt = MPI.Wtime()
grey_scale_data = image_scale_grey(img)
wt = MPI.Wtime() - wt
print('Total time taken', wt)
    
image_data = np.asarray(img)
print(image_data.shape)
#Data visualization for RGB image - frequency of scales and histogram
print('For RGB data')
visualize_data(image_data)


#Data visualization for grey scale image
print('For grey scaled data')
visualize_data(grey_scale_data)
    