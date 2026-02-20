import numpy as np 

M = 4
K = 3


matrix = np.arange(M* K)
print(matrix.shape)

matrix = matrix.reshape(M, K)
print(matrix.shape)

print(matrix)


num_SM = 4 
tile_idx = 0 


