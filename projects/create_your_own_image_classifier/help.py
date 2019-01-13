# imports

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
import json
from collections import OrderedDict
from PIL import Image
import numpy as np

# architecture choices

arch = {"vgg16":25088,
        "densenet121" :1024}

# load & transform data for models

def load_data (path_part):        
    '''
    function to load data, transform it for feeding into models
    input: part_path
    output: returns train, validation and test data 
    '''    
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])
    
    # load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)
    
    # using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    return trainloader, validationloader, testloader

# define nn architecture

def nn_architecture(architecture = 'vgg16', dropout = 0.5, fc2 = 1000, learn_r = 0.001, gpu_cpu = gpu):
     '''
    input: architecture ('vgg16' or 'densenet121'), dropout (float), fc2 (int), learn_r (float), gpu_cpu (gpu or cpu)
    output: model, critieria and optimizer
    '''
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print('please choose either vgg16 or densenet121')
        
    # define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    
    for param in model.parameters():
        param.requires_grad = False
        
        num_inputs = model.classifier[0].in_features
        
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_inputs, fc2)),
                                                ('relu1', nn.ReLU()),
                                                ('dropout1', nn.Dropout(p=dropout)),
                                                ('fc2', nn.Linear(fc2, 512)),
                                                ('relu2', nn.ReLU()),
                                                ('fc3', nn.Linear(512, 100)),
                                                ('relu3', nn.ReLU()),
                                                ('fc4', nn.Linear(100, 102)),
                                                ('output', nn.LogSoftmax(dim=1))]))
        
        model.classifier = classifier
        model.to(device = 'cuda') # 'cpu'
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = learn_r)
        
        return model, optimizer, criterion