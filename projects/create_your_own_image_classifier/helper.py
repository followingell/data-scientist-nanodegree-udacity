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

def load_data(path_part):        
	'''
	function to load data, transform it for feeding into models
	input: part_path
	output: returns train, validation and test data 
	'''    
	data_dir = path_part
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

def nn_architecture(architecture = 'vgg16', dropout = 0.5, fc2 = 1000, learn_r = 0.001, gpu_cpu = 'gpu'):
	'''
	input: architecture ('vgg16' or 'densenet121'), dropout (float), fc2 (int), learn_r (float), gpu_cpu ('gpu' or 'cpu')
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
		if gpu_cpu == 'gpu':
			model.to(device = 'cuda')
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.classifier.parameters(), lr = learn_r)
		
	return model, optimizer, criterion

# train the neural network

def train_network(model, criterion, optimizer, loader, validationloader, epoch_number = 3, progress_update = 20, gpu_cpu = 'gpu'):
	'''
	input: model, criterion, optimizer, epoch_number (int), progress_update (int), gpu_cpu ('gpu' or 'cpu') 
	output: none
	'''
	# train the classifier layers using backpropagation using the pre-trained network to get the features
	# track the loss and accuracy on the validation set to determine the best hyperparameters

	epoch_num = epoch_number
	steps = 0
	check_progress_every = progress_update
	
	for e in range(epoch_num):
		train_loss = 0
		for ii, (inputs, labels) in enumerate(loader):
			steps += 1
			if torch.cuda.is_available() and gpu_cpu == 'gpu':
				inputs, labels = inputs.to('cuda'), labels.to('cuda')
			
			optimizer.zero_grad()
			
			# forward and backward
			outputs = model.forward(inputs)
			train_loss = criterion(outputs, labels)
			train_loss.backward()
			optimizer.step()
			
			train_loss += train_loss.item()
			
			# show the model evaluation every ten steps
			
			if steps % check_progress_every == 0:
				model.eval()
				validation_loss = 0
				validation_accuracy = 0
				
				for ii, (inputs2,labels2) in enumerate(validationloader):
					optimizer.zero_grad()
					if torch.cuda.is_available() and gpu_cpu == 'gpu':
						inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
						model.to('cuda:0')
					with torch.no_grad():
						outputs = model.forward(inputs2)
						validation_loss = criterion(outputs,labels2)
						ps = torch.exp(outputs).data
						equality = (labels2.data == ps.max(1)[1])
						validation_accuracy += equality.type_as(torch.FloatTensor()).mean()
						
				validation_loss = validation_loss / len(validationloader)
				validation_accuracy = validation_accuracy / len(validationloader)
				
				print("Epoch: {}/{} ".format(e+1, epoch_num),
					  "Train Loss: {:.4f}".format(train_loss/check_progress_every),
					  "Validation Loss {:.4f}".format(validation_loss),
					  "Validation Accuracy: {:.2f}".format(validation_accuracy))
				
				train_loss = 0
				
# save the checkpoint 
def save_checkpoint(filepath = 'checkpoint.pth', architecture = 'vgg16', dropout = 0.5, learn_r = 0.001, fc2 = 1000, epoch_num = 3):
	'''
	input: epoch_num (int)
	output: none
	'''
	model.class_to_idx = train_data.class_to_idx
	model.epochs = epoch_num
	model.cpu
	
	model_checkpoint = {'structure': architecture,
						'dropout':dropout,
						'lr':learn_r,
						'fc2':fc2,
						'class_to_idx': model.class_to_idx,
						'epoch': model.epochs,
						'output_size': 102,
						'state_dict': model.state_dict(),
						'optimizer_dict': optimizer.state_dict()}
	
	torch.save(model_checkpoint, filepath)
	print ('model saved')

# load checkpoint

def load_checkpoint_rebuild_model(filepath='checkpoint.pth'):
	'''
	inputs: none
	outputs: none
	'''
	checkpoint = torch.load(filepath)
	architecture = checkpoint['structure']
	dropout = checkpoint['dropout']
	learn_r = checkpoint['lr']
	fc2 = checkpoint['fc2']
	class_to_idx = checkpoint['class_to_idx']
	num_epochs = checkpoint['epoch']
	output_size = checkpoint['output_size']
	state_dict = checkpoint['state_dict']
	optimizer_dict = checkpoint['optimizer_dict']
	
	model,_,_ = nn_architecture(architecture, dropout, fc2, learn_r)

	model.class_to_idx = checkpoint['class_to_idx']
	model.load_state_dict(checkpoint['state_dict'])
	
	print('model params loaded from checkpoint')
	
def process_image(image):
	''' 
	input: filepath
	output: numpy array
	'''
	# process a PIL image for use in a PyTorch model
	img = Image.open(image)
	transformations = transforms.Compose([transforms.Resize(256),
										  transforms.CenterCrop(224),
										  transforms.ToTensor(),
										  transforms.Normalize([0.485, 0.456, 0.406],
															   [0.229, 0.224, 0.225])])
										  
	processed_img = transformations(img)
	
	return(processed_img)

# predict image
def predict(image_path, model, topk = 5, gpu_cpu = 'gpu'):
	''' 
	input: filepath to image
	output: top predicted classes for referenced image
	'''
	# Implement the code to predict the class from an image file
	if torch.cuda.is_available() and gpu_cpu == 'gpu':
		model.to('cuda:0')
	
	img = process_image(image_path)
	img = img.unsqueeze(0)
	img = img.float()
	
	if gpu_cpu == 'gpu':
		with torch.no_grad():
			output = model.forward(img.cuda())
		
		probability = F.softmax(output.data, dim = 1)
	return probability.topk(topk)