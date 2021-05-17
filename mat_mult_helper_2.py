import numpy as np
def mat_mult_worker_2(A,B,procnum,return_dict):
    N = len(A)
    M = len(B[0])
    K = len(B)
    C = np.zeros(shape = (N,M))
    for i in range(N):
        for j in range(K):
            for k in range(M):
                C[j,k]+=(A[i,j]*B[j,k])
    return_dict[procnum] = C