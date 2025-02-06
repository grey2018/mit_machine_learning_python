# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:01:00 2020

@author: Sergiy
"""

import time
import numpy as np
import scipy.sparse as sparse

ITER = 100
K = 10
#N = 10000
N = 100

def naive(indices, k):
    mat = [[1 if i == j else 0 for j in range(k)] for i in indices]
    return np.array(mat).T


def with_sparse(indices, k):
    n = len(indices)
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
    return M


Y = np.random.randint(0, K, size=N)

# test 1
k=K
indices = Y
mat = [[1 if i == j else 0 for j in range(k)] for i in indices]
mat_t = np.array(mat).T

# test 2
n = len(indices)
M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
M_test = sparse.coo_matrix(Y, shape=(k,n)).toarray()

t0 = time.time()
for i in range(ITER):
    naive(Y, K)
print(time.time() - t0)


t0 = time.time()
for i in range(ITER):
    with_sparse(Y, K)
print(time.time() - t0)