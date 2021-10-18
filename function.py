import numpy as np
import torchvision
import torch
import torch.nn.functional as funct

from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    test_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    ])

    train_transforms = transforms.Compose([
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
    
    return trainloader, validloader, testloader

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
    classes = [labels[str(index+1)] for index in np.array(probability[1][0])]
    
    return probs, classes


