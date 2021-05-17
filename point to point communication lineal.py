#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 00:00:12 2021

@author: arailymkaiyrova
"""
from mpi4py import MPI
import os
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def send_naive(N):
    wt = MPI.Wtime()
    rand = np.zeros(N)
    if rank == 0:
        rand = np.random.random_sample(N)
        #print("Process", rank, "drew the number", rand)
        #print("Process", rank, "sending number", rand, "to process", rank+1)
        comm.Isend(rand, dest=rank+1)

    elif rank<size-1:
        #print("Process", rank, "before receiving has the number", rand)
        req = comm.Irecv(rand, source=rank-1)
        req.wait()
        #print("Process", rank, "received the number", rand, "from process", rank-1)
        
        #rand = np.random.random_sample(3)
        
        #print("Process", rank, "sending number", rand, "to process", rank+1)
        comm.Isend(rand, dest=rank+1)

    else:
        #print("Process", rank, "received the number", rand, "from process", rank-1)
        req = comm.Irecv(rand, source=rank-1)
        req.wait()
        #print("Process", rank, "received the number", rand)
    wt = MPI.Wtime() - wt
    print('Time taken to sent array size using naive approach', N, ':', wt)
        

sizes = [pow(10,3),pow(10,5),pow(10,7)]

#for size in sizes:   
#send_naive(sizes[0])
#send_naive(sizes[1])         
send_naive(sizes[2])
        

