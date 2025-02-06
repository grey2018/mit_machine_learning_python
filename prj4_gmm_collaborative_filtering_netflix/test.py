import numpy as np
import em
import common
#import naive_em

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

#X = np.loadtxt("netflix_incomplete.txt")


K = 4
n, d = X.shape
seed = 0

# TODO: Your code here

gauss_mix_init, prob_posterior_init = common.init(X, K, seed)
# INIT PARAMS
#Mu:
#[[2. 4. 5. 5. 0.]
# [3. 5. 0. 4. 3.]
# [2. 5. 4. 4. 2.]
# [0. 5. 3. 3. 3.]]
#Var: [5.93 4.87 3.99 4.51]
#P: [0.25 0.25 0.25 0.25]

prob_posterior, loglike = em.estep(X, gauss_mix_init)

gauss_mix_new = em.mstep(X, prob_posterior, gauss_mix_init)