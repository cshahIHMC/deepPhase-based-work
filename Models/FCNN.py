## Author - Chinmay Shah
# Simple FCNN model setup file


import torch
import torch.nn as nn


# Setting up a simple NN model with 31 inputs and 6 outputs
# It takes in the input of the num of layers, num of hidden units and
# dropout Rate
class FCNN(nn.Module):
    def __init__(self, inputs, outputs, numOfLayers, hiddenDimension, dropoutRate):
        super(FCNN, self).__init__()
        
        # Initilize the first layer
        self.layers = nn.ModuleList([nn.Linear(inputs, hiddenDimension)])
        # self.layers.append(nn.Dropout(p=dropoutRate))
        
        # Iterate over and add all the hidden layers
        for _ in range(numOfLayers - 1):
            self.layers.append(nn.Linear(hiddenDimension, hiddenDimension))
        
        # Add the output Layer
        self.layers.append(nn.Linear(hiddenDimension, outputs))
        
    def forward(self, x):
        
        for layer in self.layers[:-1]:
            x = x = torch.relu(layer(x))
        
        # No activation on the output layer
        x = self.layers[-1](x)
        return x
        
        