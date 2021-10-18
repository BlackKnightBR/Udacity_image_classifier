import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import shutil
import torch.nn as nn
import torch.nn.functional as funct
import function as utilidade

from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('data_directory', action = 'store',
                    help = 'Enter path to data.')

parser.add_argument('--arch', action='store',
                    dest = 'archt', default = 'vgg16',
                    help= 'Enter pretrained model to use')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.001,
                    help = 'Enter learning rate for training the model.')

parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=int, default = 0.05,
                    help = 'Enter dropout for training the model.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=int, default = 500,
                    help = 'Enter number of hidden units in classifier.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 5,
                    help = 'Enter number of epochs to use during training.')


results = parser.parse_args()
data_dir = results.data_directory
arch =  results.archt
learning_rate = float(results.lr)
dropout = float(results.drpt)
hidden_units = int(results.units)
epochs = int(results.num_epochs)

#Define the processing unit and print basic enviroment info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Processing unit:", device)
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

model = eval("models.{}(pretrained=True)".format(arch))

for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


model.classifier = classifier

trainloader, validloader, testloader = utilidade.load_data(data_dir)


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
model.to(device)

printS = 12
steps = 0


for e in range(epochs):
        running_loss = 0
        for i, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % printS == 0:
                model.eval()
                v_lost = 0
                v_accuracy=0
                for j, (inputs,labels) in enumerate(validloader):
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(device) , labels.to(device)
                    with torch.no_grad():    
                        outputs = model.forward(inputs)
                        v_lost = criterion(outputs,labels)
                        ps = torch.exp(outputs).data
                        equality = (labels.data == ps.max(1)[1])
                        v_accuracy += equality.type_as(torch.FloatTensor()).mean()
                                                                        
                v_lost = v_lost / len(validloader)
                v_accuracy = v_accuracy /len(validloader)
            
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/printS),
                  "Validation Lost {:.4f}".format(v_lost),
                   "Accuracy: {:.4f}".format(v_accuracy))
                running_loss = 0
                
                
model.eval()
model.to(device)
accuracy = 0
test_loss = 0

for i, (inputs, labels) in enumerate(testloader):
    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])

        accuracy += equality.type_as(torch.FloatTensor()).mean()
        percentage = accuracy/len(testloader)*100
print("Test Loss: {:.3f}..".format(test_loss/len(testloader)),
      "Test Accuracy: {:.2f}%".format(percentage))


save_dir = 'checkpoint.pth'

# TODO: Save the checkpoint 
torch.save(model, save_dir)

print('Training done')












