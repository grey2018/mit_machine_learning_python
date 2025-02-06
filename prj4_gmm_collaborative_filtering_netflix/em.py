"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    
    #mixture = gauss_mix_init
    
    # epsilon to avoid underflows
    eps = 1e-16
    
    n, d = X.shape
    K = mixture.mu.shape[0]
    
    #total = np.zeros((n, K))
    #posterior = np.zeros((n, K))

    log_total = np.zeros((n, K))
    log_posterior = np.zeros((n, K))
       
    #j=0
    for j in range(K):
         
        #u=0
        for u in range(n):
            
            d_c = np.count_nonzero(X[u]) # length of C[u]
            C = np.asarray(np.nonzero(X[u])).flatten() # indices of X[u]
            #d_c2 = C.size # check => should be the same as dd
            X_c = np.zeros((d_c))
            mu_c = np.zeros((d_c))

            #first = 1 / (2 * np.pi * mixture.var[j])**(d_c/2)
            log_first = (d_c/2) * np.log(2 * np.pi * mixture.var[j])            
            
            for i_c in range(d_c):
                X_c[i_c] = X[u, C[i_c]]
                mu_c[i_c] = mixture.mu[j, C[i_c]]

            dist = X_c - mu_c
            second = (-1 * np.linalg.norm(dist)**2)/(2*mixture.var[j])
            #gauss_pdf = first * np.exp(second)
            
            #total[u,j] = mixture.p[j] * gauss_pdf
            #log_total[u,j] = np.log(mixture.p[j]) + np.log(gauss_pdf)
            #log_total[u,j] = np.log(mixture.p[j]) - log_first + second
            log_total[u,j] = np.log(mixture.p[j] + eps) - log_first + second
    
    #total_sum = np.sum(total, axis=1)
    log_total_sum = logsumexp(log_total, axis=1)
    #total_sum_2 = np.exp(log_total_sum)

    for j in range(K):    
        #posterior[:,j] = total[:,j] / total_sum
        log_posterior[:,j] = log_total[:,j] - log_total_sum
    
    posterior_2 = np.exp(log_posterior)

    #log_array = np.log(total_sum)
    #log_array = np.log(total_sum_2)
    #log_like = np.sum(np.log(total_sum_2))
    log_like = np.sum(log_total_sum)
    
    return posterior_2, log_like
    
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    #post = prob_posterior
    #mixture = gauss_mix_init
    #min_variance = .25
    
    n, d = X.shape
    K = post.shape[1]
    mu = np.zeros((K, d))
    var = np.zeros(K)
    delta = np.zeros((n, d))
    
    posterior_sum = np.sum(post, axis=0)
    prior = posterior_sum / n
    
    posterior_marg = np.zeros((n,K,d))
    
    #posterior_sum_clusters = np.sum(post, axis=1)
    
    ##################################
    # mu part
    nom_mu = np.zeros((n, K, d))
    for u in range(n):
        #d_c = np.count_nonzero(X[u]) # length of C[u]
        for j in range(K):
            for i in range(d): 
                nom_mu[u,j,i] = post[u,j] * X[u,i]
                if X[u, i] != 0:
                    delta[u, i] = 1
                    posterior_marg[u,j,i] = post[u,j]
                else:
                    posterior_marg[u,j,i] = 0
    
    posterior_marg_sum = np.sum(posterior_marg, axis=0)         
 
    for j in range(K):
        for i in range(d):
            nom_mu_sum = np.sum(nom_mu[:,j,i], axis=0)
            if posterior_marg_sum[j,i] >= 1:
               #mu[j,i] = np.sum(nom_mu[:,j,i], axis=0) / posterior_sum[j]
               mu[j,i] = nom_mu_sum / posterior_marg_sum[j,i]
            else:
               mu[j,i] = mixture.mu[j,i]
                   
    #########################################
    # var part
    nom_var = np.zeros((n, K))
    denom_var = np.zeros((n, K))
    dist = np.zeros((n,K,d))
    
    for u in range(n):
        d_c = np.count_nonzero(X[u]) # length of C[u]
        #C = np.asarray(np.nonzero(X[u])).flatten() # indices of X[u]
        for j in range(K):
            #denom_var[u, j] = d_c * posterior_sum[j]
            denom_var[u, j] = d_c * post[u, j]
            # TODO: norm only for observed movies
            for i in range(d):
                if X[u,i] == 0:
                    dist[u,j,i] = 0
                else:
                    dist[u,j,i] = X[u,i] - mu[j,i]
            #nom_var[u, j] = post[u,j] * np.linalg.norm((X[u] - mu[j]))**2
            nom_var[u, j] = post[u,j] * np.linalg.norm(dist[u,j])**2
    
    #var = np.sum(nom_var, axis=0) @ (1. / (d * posterior_sum))
    nom_var_sum = np.sum(nom_var, axis=0)
    denom_var_sum = np.sum(denom_var, axis=0)
    var_orig = np.multiply(nom_var_sum, (1. / denom_var_sum ))
    var = np.maximum(var_orig, min_variance)
    
    gauss_mix = GaussianMixture(mu, var, prior)
    
    return gauss_mix
    
    
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # rule: apply until: new log-likelihood−old log-likelihood≤10−6⋅|new log-likelihood|
    eps = 1e-6
    prev_loglike = None
    loglike = None
    iteration = 0
    while (prev_loglike is None or loglike - prev_loglike > eps * abs(loglike)):
        iteration += 1
        prev_loglike = loglike
        post, loglike = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    print("iterations=", iteration)
    return mixture, post, loglike
    
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    ##############################################
    # grey2018: re-use of estep
    ##############################################
    
    #mixture = gauss_mix
    
    # epsilon to avoid underflows
    eps = 1e-16
    
    n, d = X.shape
    K = mixture.mu.shape[0]
    
    #total = np.zeros((n, K))
    #posterior = np.zeros((n, K))

    log_total = np.zeros((n, K))
    log_posterior = np.zeros((n, K))
       
    #j=0
    for j in range(K):
         
        #u=0
        for u in range(n):
            
            d_c = np.count_nonzero(X[u]) # length of C[u]
            C = np.asarray(np.nonzero(X[u])).flatten() # indices of X[u]
            #d_c2 = C.size # check => should be the same as dd
            X_c = np.zeros((d_c))
            mu_c = np.zeros((d_c))

            #first = 1 / (2 * np.pi * mixture.var[j])**(d_c/2)
            log_first = (d_c/2) * np.log(2 * np.pi * mixture.var[j])            
            
            for i_c in range(d_c):
                X_c[i_c] = X[u, C[i_c]]
                mu_c[i_c] = mixture.mu[j, C[i_c]]

            dist = X_c - mu_c
            second = (-1 * np.linalg.norm(dist)**2)/(2*mixture.var[j])
            #gauss_pdf = first * np.exp(second)
            
            #total[u,j] = mixture.p[j] * gauss_pdf
            #log_total[u,j] = np.log(mixture.p[j]) + np.log(gauss_pdf)
            #log_total[u,j] = np.log(mixture.p[j]) - log_first + second
            log_total[u,j] = np.log(mixture.p[j] + eps) - log_first + second
    
    #total_sum = np.sum(total, axis=1)
    log_total_sum = logsumexp(log_total, axis=1)
    #total_sum_2 = np.exp(log_total_sum)

    for j in range(K):    
        #posterior[:,j] = total[:,j] / total_sum
        log_posterior[:,j] = log_total[:,j] - log_total_sum
    
    posterior_2 = np.exp(log_posterior)
    
    ##############################################
    # grey2018: end of re-use of estep
    ##############################################
    
    #X_pred = X # grey2018: WRONG! such an assignment will alter also X if X_pred is changed
    X_pred = np.copy(X)
    
    #u=0
    for u in range(n):
        #i=0
        for i in range(d):
            if X_pred[u,i] == 0:
                X_pred[u,i] = posterior_2[u] @ mixture.mu[:,i]
                
    return X_pred
    raise NotImplementedError
