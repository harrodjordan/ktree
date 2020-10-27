import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
from torch.optim.optimizer import required
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from pytorchtools import EarlyStopping



def train_test_ktree(model, trainloader, validloader, testloader, epochs=10, randorder=False, patience=60):
    '''
    Trains and tests k-tree models
    Inputs: model, trainloader, validloader, testloader, epochs, randorder, patience
    Outputs: train loss_curve, train acc_curve, test ave_loss, test accuracy, trained model
    '''
    # Initialize loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # to track training loss and accuracy as model trains
    loss_curve = []
    acc_curve = []
    
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    # if randorder == True, generate the randomizer index array for randomizing the input image pixel order
    if randorder == True:
        ordering = torch.randperm(len(trainloader.dataset.tensors[0][0]))
    
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    for epoch in range(epochs):  # loop over the dataset multiple times
        ######################    
        # train the model    #
        ######################
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                inputs = inputs[:,ordering]
            else:
                inputs = inputs

            labels = labels

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().reshape(-1,1))
            loss.backward()
            
####        # Freeze select weights by zeroing out gradients
            for child in model.children():
                for param in child.parameters():
                    for freeze_mask in model.freeze_mask_set:
                        if param.grad.shape == freeze_mask.shape:
                            param.grad[freeze_mask] = 0
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += (torch.round(outputs) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size
            # Generate loss and accuracy curves by saving average every 4th minibatch
            if (i % 4) == 3:    
                loss_curve.append(running_loss/4)
                acc_curve.append(running_acc/4)
                running_loss = 0.0
                running_acc = 0.0
        
    
    print('Finished Training, %d epochs' % (epoch+1))
    

    if randorder == True:
        return(loss_curve, acc_curve, model, ordering)
    else:
        return(loss_curve, acc_curve, model)

def train_test_fc(model, trainloader, validloader, testloader, epochs=10, patience=60):
    '''
    Trains and tests fcnn models
    Inputs: model, trainloader, validloader, testloader, epochs, patience
    Outputs: train loss_curve, train acc_curve, test ave_loss, test accuracy, trained model
    '''
    # Initialize loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # to track training loss and accuracy as model trains
    loss_curve = []
    acc_curve = []
    
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)
        
        
    for epoch in range(epochs):  # loop over the dataset multiple times
        ######################    
        # train the model    #
        ######################
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs = inputs
            labels = labels

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().reshape(-1,1))
            loss.backward()
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += (torch.round(outputs) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size
            if i % 4 == 3:      # Generate loss and accuracy curves by saving average every 4th minibatch
                loss_curve.append(running_loss/4)
                acc_curve.append(running_acc/4)
                running_loss = 0.0
                running_acc = 0.0
    
    print('Finished Training, %d epochs' % (epoch+1))
        
    return(loss_curve, acc_curve, model)
