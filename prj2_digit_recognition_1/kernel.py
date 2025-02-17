import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    
    return (X @ Y.T + c)**p
    
    raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    row_x, col_x = X.shape
    row_y, col_y = Y.shape
    norm_matrix = np.empty((row_x, row_y))
    
    # expansion: NORM(X-Y)^2 = NORM(X)^2 - 2* X'Y + NORM(Y)^2
    for i in range(row_x):
        for j in range(row_y):
            norm_matrix[i,j] = np.linalg.norm(X[i])**2 + np.linalg.norm(Y[j])**2 - 2 * X[i] @ Y[j].T
            # solution by MIT staff:
            #norm_matrix[i,j] = np.linalg.norm(X[i] - Y[j]) ** 2
    
    kernel = np.exp(-gamma * (norm_matrix))
    return kernel
    raise NotImplementedError
