def add_without_shared(A,B,C):
    for i in range(len(A)):
        C[i] = A[i]+B[i]
    
def add_with_shared(A,B,procnum,return_dict):
    C = []
    for i in range(len(A)):
        C.append(A[i]+B[i])
    return_dict[procnum] = C
    