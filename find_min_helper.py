def find_minimum(A,q):
    N = len(A)
    curr_min = A[0]
    for i in range(1,N):
        curr_min = min(curr_min,A[i])
    q.put(curr_min)