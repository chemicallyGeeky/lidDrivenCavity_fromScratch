#Gauss Siedel solver that uses sparse A and solves Ax = b
import numpy as np
from numba import njit

def gaussSiedel_solver(A, b, X0=None, nIter=100, tol=1e-6):
    
    # convert A_sparse to A
    #A = A_sparse.toarray()
    
    #initialize:
    if X0 is None:
        X = np.zeros(len(b))
    else:
        X = X0.copy()   
    
    return gaussSiedel_core(A, b, X, nIter, tol)
      

@njit
def gaussSiedel_core(A, b, X, nIter, tol):
    N = len(A)
            
    for iter in range(nIter):
        max_diff = 0.0 # tracks the maximum cell change
        
        for i in range(N):
            old_val = X[i]
            sum_val = 0
            for k in range(N):
                if k != i:
                    if A[i, k] != 0.0:
                        sum_val += A[i, k]*X[k]
            X[i] = (b[i] - sum_val)/A[i,i]  
            diff = abs(X[i] - old_val)
            if diff > max_diff:
                max_diff = diff
        
        #residual 
        #res = np.linalg.norm(X-X_orig)

        #exit loop
        if max_diff < tol:
            return X, iter+1    
    
    return X, nIter  