import numpy as np
import kmeans
import common
#import naive_em
import em

X = np.loadtxt("toy_data.txt")
X = np.loadtxt("test_complete.txt")
X = np.loadtxt("test_incomplete.txt")

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")



# TODO: Your code here

# k-means: different Ks and seeds

#n, d = X.shape
#K = np.asarray([1, 2, 3, 4])
#seed = np.asarray([0, 1, 2, 3, 4])
#kSize = K.size
#seedSize = seed.size
#cost = np.zeros((kSize, seedSize))
#
#for k in range(0, kSize):
#    print("k=", K[k])
#    
#    for s in range(0, seedSize):
#        print("seed=", seed[s])
#
#        GM, post = common.init(X=X, K=K[k], seed=seed[s])
#        GM_best, post, cost[k][s] = kmeans.run(X, GM, post)
#
#        #title = “K= %i, seed= %i” % (K[k], seed[s])
#        common.plot(X, GM_best, post, "title")
#
#print("cost=", cost)
#mincost = np.amin(cost, axis=1)
#print("min cost=", mincost)

#################################################
# test naive Exp-Max Algorithm

# Using K=3 and a seed of 0, on the toy dataset, you should get a log likelihood of -1388.0818. 
#n, d = X.shape # users x movie_features

#K = 3 # 
#seed = 0
#LL_target = -1388.0818
#
#gauss_mix, post = common.init(X, K=K, seed=seed)
#
##prob_posterior, LL_new = naive_em.estep(X, gauss_mix)
##gauss_mix_new = naive_em.mstep(X, prob_posterior)
#
#gauss_mix_new, post_new, LL_new = naive_em.run(X, gauss_mix, post)

###################################################
# Exp-Max Algorithm: different Ks and seeds

#n, d = X.shape
#K = np.asarray([1, 2, 3, 4])
K = np.asarray([12])
#seed = np.asarray([2, 3, 4, 0, 1])
seed = np.asarray([1])

kSize = K.size
seedSize = seed.size
loglike = np.zeros((kSize, seedSize))
bic = np.zeros((kSize, seedSize))

for k in range(0, kSize):
    print("k=", K[k])
    
    for s in range(0, seedSize):
        print("seed=", seed[s])
        
        gauss_mix_init, prob_posterior_init = common.init(X=X, K=K[k], seed=seed[s])
        #gauss_mix, prob_posterior, loglike[k][s] = naive_em.run(X, gauss_mix_init, prob_posterior_init)
        gauss_mix, prob_posterior, loglike[k][s] = em.run(X, gauss_mix_init, prob_posterior_init)
        bic[k,s] = common.bic(X, gauss_mix, loglike[k,s]) # if based on the MAX loglike, can be placed after the loop
        
        print(gauss_mix)
        print(loglike[k][s])
        #title = “K= %i, seed= %i” % (K[k], seed[s])
        #common.plot(X, gauss_mix, prob_posterior, "title")
        
##########################################
# DETERMINE THE BEST CONSTELLATION
print("loglike=", loglike)
#mincost = np.amin(cost, axis=1)
max_loglike = np.amax(loglike, axis=1)
print("max loglike=", max_loglike)

max_bic = np.amax(bic, axis=1)
print("max bic=", max_bic)

#############################################
# PREDICT AND BACK-TEST
# use the best gauss mix (and posteriors) for prediction of missing values
X_pred = em.fill_matrix(X, gauss_mix)
RMSE = common.rmse(X_gold, X_pred)
