import numpy as np
import torchvision
import torch
import torch.nn.functional as funct
import torch.nn as nn

from collections import OrderedDict
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),                    
        transforms.CenterCrop(224),                          
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(256),                                     
        transforms.CenterCrop(224),                                         
        transforms.ToTensor(),                                      
        transforms.Normalize([0.485, 0.456, 0.406],                                                                
                             [0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size= 32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
    
    return trainloader, validloader, testloader, train_data

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    treated_img = Image.open(image)
    img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pymodel_img = img(treated_img)
    
    return pymodel_img

def predict(image, model, topk, labels, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    img_torch = process_image(image)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.to(device))
        
    probability = funct.softmax(output.data,dim=1)
    probability = probability.topk(topk)
    probs = np.array(probability[0][0])
    classes = [idx2class[i] for i in ps_top[1].tolist()[0]]
    
    return probs, classes

def create_model(arch, hidden1, hidden2, dropout, nClasses):
    model = eval("models.{}(pretrained=True)".format(arch))

    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(hidden1,hidden2)),
                            ('relu1', nn.ReLU()),
                            ('d_out1',nn.Dropout(dropout)),
                            ('fc2', nn.Linear(hidden2, 1024)),
                            ('relu2', nn.ReLU()),
                            ('d_out2',nn.Dropout(p=dropout)),
                            ('fc3', nn.Linear(1024, nClasses)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier        
    return model

def save_model(model, train_data, arch, dropout):
    
    model.class_to_idx = train_data.class_to_idx
    torch.save({'structure':arch,
                'hidden1': hidden1,
                'hidden2': hidden2,
                'dropout': dropout,
                'nclasses': nClasses,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                'checkpoint.pth')

    return True

def load_model(path):
       
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden1 = checkpoint['hidden1']
    hidden1 = checkpoint['hidden2']
    dropout = chekpoint['dropout']
    nClasses = checkpoint['nClasses']
    model = create_model(structure , hidden1, hidden2, dropout, nClasses)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
