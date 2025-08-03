## Author - Chinmay Shah
# Simple LSTM model setup file


import torch
import torch.nn as nn

# Setting up a simple LSTM model 
class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, outputSize):
        super(LSTM, self).__init__()
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True)
        # Define the fully connected output layer
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
         
        # Get RNN outputs and hidden states
            # x: (batch_size, seq_length, input_size)
            # out: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)
        
        # Apply fully connected layer to the output from the last time step
            # out: (batch_size, hidden_size)
            # out: (batch_size, output_size) - from the fully connected layer
        out = self.fc(out[:, -1, :])
    
        # out: (batch_size, 1, output_size)
        out = out.unsqueeze(1)
        
        return out
    