import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE
    
#    n, d, k = 3, 5, 7
#    X = np.arange(0, n * d).reshape(n, d)
#    theta = np.zeros((k, d))
#    temp_parameter = 0.2
#    exp_res = np.ones((k, n)) / k

    #X_temper = X / temp_parameter # remains [n examples x d features]
    #product = np.matmul(theta, np.transpose(X_temper)) # [k classes x n examples]
    product = np.matmul(theta, np.transpose(X / temp_parameter)) # [k classes x n examples]
    C = np.amax(product, axis=0) # max over rows = vector [n examples]
    
    #product_adj = product - C # [k classes x n examples] minus [n examples]
    #product_expon = np.exp(product_adj) # remains [k classes x n examples]
    
    product_expon = np.exp(product - C) # remains [k classes x n examples]
    total = 1 / np.sum(product_expon, axis=0) # sum over rows = vector [n examples]
    
    #total_matrix = np.diag(total) # [n examples x n examples]
    # todo: use the sparse matrix
    #total_matrix = sparse.spdiags(total, 0, total.size, total.size).toarray()
    # todo: use loop (slowlier but maybe will not produce Memory Error)
    
    #H = np.matmul(product_expon, total_matrix) # [k classes x n examples]
    #H = product_expon @ total_matrix # [k classes x n examples]
    
    H = np.empty(product_expon.shape) # [k classes x n examples]
    for i in range(total.size):
        H.T[i] = product_expon.T[i] * total[i]
    
    
    #return product_expon @ total_matrix # [k classes x n examples]
    return H
    
    raise NotImplementedError

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    
#    n, d, k = 3, 5, 7
#    X = np.arange(0, n * d).reshape(n, d)
#    Y = np.arange(0, n)
#    theta = np.zeros((k, d))
#    temp_parameter = 0.2
#    lambda_factor = 0.5
#    exp_res = 1.9459101490553135
    
    n = X.shape[0] # number of examples
    #d = X.shape[1] # number of features
    k = theta.shape[0] # number of classes (digits)
        
    prob_exp = compute_probabilities(X, theta, temp_parameter)
    prob_log = np.where (prob_exp != 0, np.log(prob_exp), 0) # k classes x n examples
    
    digits = np.arange(k)
    
    class_as_label = np.zeros((k, n)) # k classes x n examples
    for i in range(n):
        #np.transpose(class_as_label)[i] = np.equal(digits, Y[i])
        class_as_label.T[i] = np.equal(digits, Y[i])
    
    loss = -1.0 / n * np.sum(np.multiply(class_as_label, prob_log))
    #regularization = lambda_factor / 2 * np.sum(np.multiply(theta, theta))
    regularization = lambda_factor / 2 * np.sum(theta**2)
    cost = loss + regularization
    
    return cost
       
    raise NotImplementedError

def run_gradient_descent_iteration_grey2018(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    
#    n, d, k = 3, 5, 7
#    X = np.arange(0, n * d).reshape(n, d)
#    Y = np.arange(0, n)
#    theta = np.zeros((k, d))
#    alpha = 2
#    temp_parameter = 0.2
#    lambda_factor = 0.5
#    exp_res = np.zeros((k, d))
#    exp_res = np.array([
#       [ -7.14285714,  -5.23809524,  -3.33333333,  -1.42857143, 0.47619048],
#       [  9.52380952,  11.42857143,  13.33333333,  15.23809524, 17.14285714],
#       [ 26.19047619,  28.0952381 ,  30.        ,  31.9047619 , 33.80952381],
#       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
#       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
#       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
#       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286]
#    ])
    
    # compute gradient for given theta
    n = X.shape[0] # number of examples
    #d = X.shape[1] # number of features
    k = theta.shape[0] # number of classes (digits)
    digits = np.arange(k)
    
    class_as_label = np.zeros((k, n)) # k classes x n examples
    for i in range(n):
        #np.transpose(class_as_label)[i] = np.equal(digits, Y[i])
        class_as_label.T[i] = np.equal(digits, Y[i])
    
    prob_exp = compute_probabilities(X, theta, temp_parameter)
    
    prob_delta = class_as_label - prob_exp
    impact_delta = prob_delta @ X
    loss_delta = -1.0 / (temp_parameter * n) * impact_delta
    reg_delta = lambda_factor * theta
    
    gradient = loss_delta + reg_delta
    theta_new = theta - alpha * gradient
    
    return theta_new
    
    raise NotImplementedError

# more elegant solution by MIT staff    
def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    
    itemp=1./temp_parameter
    num_examples = X.shape[0]
    num_labels = theta.shape[0]
    probabilities = compute_probabilities(X, theta, temp_parameter)
    # M[i][j] = 1 if y^(j) = i and 0 otherwise.
    M = sparse.coo_matrix(([1]*num_examples, (Y,range(num_examples))), shape=(num_labels,num_examples)).toarray()
    non_regularized_gradient = np.dot(M-probabilities, X)
    non_regularized_gradient *= -itemp/num_examples
    
    return theta - alpha * (non_regularized_gradient + lambda_factor * theta)
    
    raise NotImplementedError    
    

    

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    
    return (train_y % 3, test_y % 3)
    
    raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    assigned_labels = get_classification(X, theta, temp_parameter) % 3
    return 1 - np.mean(assigned_labels == Y)
    
    raise NotImplementedError

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
