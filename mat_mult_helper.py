import numpy as np
def mat_mult_worker(A,B,C):
    N = len(A)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[j,k]+=(A[i,j]*B[j,k])
    print(C)
    