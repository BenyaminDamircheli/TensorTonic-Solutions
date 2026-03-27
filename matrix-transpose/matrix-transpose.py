import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    n, m = A.shape

    AT = np.zeros((m, n))

    for x in range(m):
        for y in range(n):
            AT[x][y] = A[y][x]
    
    return AT
        
