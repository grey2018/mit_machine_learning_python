import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    # TODO
    return np.maximum(0, x)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    # TODO
    return 1 if x > 0 else 0

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1') # 3 by 2
        self.hidden_to_output_weights = np.matrix('1 1 1') # 1 by 3
        self.biases = np.matrix('0; 0; 0') # 3 by 1
        self.learning_rate = .001
        self.epochs_to_train = 10
        
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]
        
        # tran 1:
        #self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        # train 2:
        self.training_points =  [((2,-4),-2), ((7,3),10), ((-7,5),-2), ((-4,-4),-8), ((-5,-3),-8),
                                 ((0,4),4), ((-5,7),2), ((-9,9),0), ((-1,-8),-9), ((8,2),10)]



    def train(self, x1, x2, y):

        ##################################
        # vectorization
        relu_vect = np.vectorize(rectified_linear_unit)
        relu_der_vect = np.vectorize(rectified_linear_unit_derivative)
        
        ###################################
        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1
                
        ##################################
        # np.array instead of np.matrix!!!
        # be aware of the shapes (some inputs are mirrorred)!!!
        arr_input_to_hidden_weights = np.array(self.input_to_hidden_weights).astype('float64')
        arr_hidden_to_output_weights = np.array(self.hidden_to_output_weights).astype('float64')
        arr_biases = np.array(self.biases).astype('float64')
        arr_input_values = np.array(input_values).astype('float64') 

        # Calculate the input and activation of the hidden layer
        # TODO (3 by 1 matrix)
        # np.matrix uses * for matmul, np.array uses @ for matmul and * for elemtwise multiply
        hidden_layer_weighted_input = arr_input_to_hidden_weights @ arr_input_values 
        hidden_layer_weighted_input_bias = hidden_layer_weighted_input + arr_biases
        #print('Z (HL weighted Input: ', hidden_layer_weighted_input)
        # TODO (3 by 1 matrix)
        hidden_layer_activation = relu_vect(hidden_layer_weighted_input_bias)
        #print('A (HL activation: ', hidden_layer_activation)
        
        # TODO
        output =  (arr_hidden_to_output_weights @ hidden_layer_activation)[0]
        #print('Output: ', output)
        # TODO
        activated_output = output_layer_activation(output)
        #print('Activated Output: ', activated_output)
        
        
        ####################################
        ### Backpropagation ###

        # Compute gradients
        
        # TODO
        #loss = 0.5 * (y - activated_output)**2.
        #print('Loss: ', loss)
        # = delta loss wrt output * derivative of output activation
        #output_layer_error = (activated_output - y) * output_layer_activation_derivative(activated_output)
        output_layer_error = (activated_output - y) * output_layer_activation_derivative(output)
        #output_layer_error = (activated_output - y) # scalar
        #print('OL_error type: ', type(output_layer_error))
        #print('OL_error: ', output_layer_error)
        
        # TODO (3 by 1 matrix)
        # = (weight * error) * derivative of hidden activation
        HLA_deriv = relu_der_vect(hidden_layer_weighted_input_bias)
        #print('HLA_deriv: ', HLA_deriv)
                
        HLE_step1 =  (arr_hidden_to_output_weights * output_layer_error).T
        #print('HLE_step1: ', HLE_step1)

        # element-wise multiplication needed!
        hidden_layer_error = np.multiply(HLE_step1, HLA_deriv)
        #print('HL_error: ', hidden_layer_error)        
                        
        # TODO
        bias_gradients = hidden_layer_error * 1.
        #print('bias_gradients: ', bias_gradients)
        
        # TODO
        #hidden_to_output_weight_gradients = hidden_layer_weighted_input_bias * output_layer_error
        hidden_to_output_weight_gradients = hidden_layer_activation * output_layer_error
        #hidden_to_output_weight_gradients = np.multiply(hidden_layer_weighted_input, hidden_layer_error)
        #hidden_to_output_weight_gradients = hidden_layer_weighted_input_nob * output_layer_error
        #print('weight_HTO_gradients: ', hidden_to_output_weight_gradients)
        
        # TODO
        input_to_hidden_weight_gradients =  hidden_layer_error * arr_input_values.T
        #print('weight_ITH_gradients: ', input_to_hidden_weight_gradients)
        
        # Use gradients to adjust weights and biases using gradient descent
        # TODO
        arr_biases = arr_biases - self.learning_rate * bias_gradients
        # TODO
        arr_input_to_hidden_weights = arr_input_to_hidden_weights - self.learning_rate * input_to_hidden_weight_gradients
        # TODO
        arr_hidden_to_output_weights = arr_hidden_to_output_weights - self.learning_rate * hidden_to_output_weight_gradients.T
        
        self.biases = np.matrix(arr_biases)
        self.input_to_hidden_weights = np.matrix(arr_input_to_hidden_weights)
        self.hidden_to_output_weights =  np.matrix(arr_hidden_to_output_weights)
        
    def predict(self, x1, x2):

        input_values = np.matrix([[x1],[x2]])
                
        arr_input_to_hidden_weights = np.array(self.input_to_hidden_weights).astype('float64')
        arr_hidden_to_output_weights = np.array(self.hidden_to_output_weights).astype('float64')
        arr_biases = np.array(self.biases).astype('float64')
        arr_input_values = np.array(input_values).astype('float64') 
        
        ##################################
        # vectorization
        relu_vect = np.vectorize(rectified_linear_unit)
        relu_der_vect = np.vectorize(rectified_linear_unit_derivative)

        # Compute output for a single input(should be same as the forward propagation in training)
        #hidden_layer_weighted_input = # TODO
        #hidden_layer_activation = # TODO
        
        hidden_layer_weighted_input = arr_input_to_hidden_weights @ arr_input_values 
        hidden_layer_weighted_input_bias = hidden_layer_weighted_input + arr_biases
        hidden_layer_activation = relu_vect(hidden_layer_weighted_input_bias)
        
        #output = # TODO
        #activated_output = # TODO
        
        #output =  float(arr_hidden_to_output_weights @ hidden_layer_activation)
        output =  arr_hidden_to_output_weights @ hidden_layer_activation
        activated_output = output_layer_activation(output)

        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):
        
        # grey2018: debug
        print('Training pairs:  ', self.training_points)
        print('Starting params:  ')
        print('')
        print('(Input --> Hidden Layer) Weights:  ', self.input_to_hidden_weights)
        print('(Hidden --> Output Layer) Weights:  ', self.hidden_to_output_weights)
        print('Biases:  ', self.biases)

        for epoch in range(self.epochs_to_train):
            print('')
            print('Epoch  ', epoch)            
            for x,y in self.training_points:
                self.train(x[0], x[1], y)
                if epoch == 0:
                    print('(Input --> Hidden Layer) Weights:  ', self.input_to_hidden_weights)
                    print('(Hidden --> Output Layer) Weights:  ', self.hidden_to_output_weights)
                    print('Biases:  ', self.biases)
#            if epoch == 0:
#                print('Epoch END')
#                print('(Input --> Hidden Layer) Weights:  ', self.input_to_hidden_weights)
#                print('(Hidden --> Output Layer) Weights:  ', self.hidden_to_output_weights)
#                print('Biases:  ', self.biases)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            print("Point,", point, "Expectation,", 7*point[0])
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
