import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

class MLP(nn.Module):

    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        
        # TODO initialize model layers here
        #self.input_dimension = input_dimension
        
        #self.length_single = 0        
        self.linear_hid = nn.Linear(input_dimension, 64)
        self.linear_out = nn.Linear(64, 20)
               

    def forward(self, x):
        
        # X dimensions: batchsize x height x width
        #self.length_single = int(self.input_dimension / x.shape(1) * x.shape(2))
        
        
        xf = self.flatten(x) # XF dimensions: batchsize x (height * width)
        
        #print("Flattened X", xf)
        #print("Dimensions XF", xf.size())
        
        #xf1 = xf.narrow(1, 0, self.length_single)
        #xf2 = xf.narrow(1, self.input_dimension - self.length_single - 1, self.length_single)
        
        #xf1 = xf[0:self.length_single]
        #xf2 = xf[(self.input_dimension-self.length_single-1):self.input_dimension]
        
        ##########################
        
        #hid1 = F.relu(self.linear_hid(xf1))
        #out1 = F.relu(self.linear_out(hid1))
        
        #hid2 = F.relu(self.linear_hid(xf2))
        #out2 = F.relu(self.linear_out(hid2))
        
        hid = self.linear_hid(xf)
        out = self.linear_out(hid)
        

        
        #print("Output: ", out)
        #print("Dimensions: ", out.size())
                
        # dimensions: batchsize x output_neurons = 64 x 20
        # divide the second dimension (i.e. dim=1) in 2 chunks
        out_chunk = torch.chunk(out, 2, dim=1)
        out_first_digit = out_chunk[0]
        out_second_digit = out_chunk[1]
        
        #print("1st: ", out_first_digit)
        #print("1st Dimensions: ", out_first_digit.size())
      
        # TODO use model layers to predict the two digits
        
        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = MLP(input_dimension) # TODO add proper layers to MLP class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
