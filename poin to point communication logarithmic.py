from mpi4py import MPI
import os
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


receiver_id = (rank-2)//2 if rank%2==0 else (rank-1)//2
sender_left, sender_right = 2*rank+1, 2*rank+2
def send_data(N):
    wt = MPI.Wtime()
    if rank == 0:
        rand = np.random.random_sample(N)
        #print("Process", rank, "drew the number", rand)
        #print("Process", rank, "sending number", rand, "to process", sender_left)
        comm.Isend(rand, dest=sender_left)
        #print("Process", rank, "sending number", rand, "to process", sender_right)
        comm.Isend(rand, dest=sender_right)

    else:
        rand = np.empty(N)
        req = comm.Irecv(rand, source=receiver_id )
        req.wait()
        #print("Process", rank, "received the number", rand, "from process", receiver_id)
    
        if sender_left<size:
            #print("Process", rank, "sending number", rand, "to process", sender_left)
            comm.Isend(rand, dest=sender_left)

        if sender_right<size:
            #print("Process", rank, "sending number", rand, "to process", sender_right)
            comm.Isend(rand, dest=sender_right)
        wt = MPI.Wtime() - wt
        print('Time taken to sent array size using efficient approach', N, ':', wt)

sizes = [pow(10,3),pow(10,5),pow(10,7)]

#for size in sizes:   
#send_data(sizes[0])
#send_data(sizes[1])         
send_data(sizes[2])




