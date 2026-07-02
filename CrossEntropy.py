import torch
import torch.nn as nn
import torch.optim as optim

class CrossEntropyModel(nn.Module):
    def __init__(self,cascade_arr,lr):
        super().__init__()
        #initialize layer
        layers = []
        for i in range(len(cascade_arr)-1):
            layers.append(nn.Linear(cascade_arr[i],cascade_arr[i+1]))
            if i < len(cascade_arr) - 2:  #add relu in between each layer except the last
                layers.append(nn.ReLU())
        
        #sequence layers
        self.network = nn.Sequential(*layers)

        #defin loss fuction 
        self.loss_fn = nn.CrossEntropyLoss()

        #define optimizer function 
        self.optimizer = optim.Adam(self.parameters(),lr = lr)

    def forward(self,x):
        return self.network(x)
    
    def backprop(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




