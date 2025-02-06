"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    
    # dim(mu) = K x d
    # dim(var) = K
    
    #mixture = gauss_mix
    
    n, d = X.shape
    K = mixture.mu.shape[0]
    posterior = np.zeros((n, K))
    total = np.zeros((n, K))
    
    #i=0
    for i in range(K):
    
        #first = 1 / (2*np.pi * mixture.var[i])**(1/2)
        first = 1 / (2*np.pi * mixture.var[i])**(d/2)
        dist = X - mixture.mu[i]
        
        #j=0
        for j in range(n):
            second = (-1 * np.linalg.norm(dist[j])**2)/(2*mixture.var[i])
            gauss_pdf = first * np.exp(second)
            total[j,i] = mixture.p[i] * gauss_pdf
    
    total_sum = np.sum(total, axis=1) 

    for i in range(K):    
        posterior[:,i] = total[:,i] / total_sum
        
    #log_array = np.log(posterior)
    log_array = np.log(total_sum)
    log_like = np.sum(log_array)
    
    return posterior, log_like
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    #post = prob_posterior
    
    n, d = X.shape
    K = post.shape[1]
    mu = np.zeros((K, d))
    var = np.zeros(K)
    
    posterior_sum = np.sum(post, axis=0)
    prior = posterior_sum / n
    
    nom_mu = np.zeros((n, K, d))
    for i in range(n):
        for j in range(K):
            for k in range(d):
                nom_mu[i, j, k] = post[i,j] * X[i][k]
    
    for j in range(K):
        for k in range(d):
               mu[j][k] = np.sum(nom_mu[:,j,k], axis=0) / posterior_sum[j]   
 
    nom_var = np.zeros((n, K))
    for i in range(n):
        for j in range(K):
            nom_var[i, j] = post[i,j] * np.linalg.norm((X[i] - mu[j]))**2          
    
    #var = np.sum(nom_var, axis=0) @ (1. / (d * posterior_sum))
    var = np.multiply(np.sum(nom_var, axis=0), (1. / (d * posterior_sum)))
    
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
        mixture = mstep(X, post)

    #print(iteration)
    return mixture, post, loglike
    
    raise NotImplementedError
