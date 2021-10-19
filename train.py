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
                    dest='drpt', type=float, default = 0.05,
                    help = 'Enter dropout for training the model.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=float, default = 500,
                    help = 'Enter number of hidden units in classifier.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 1,
                    help = 'Enter number of epochs to use during training.')

parser.add_argument('--device', action = 'store',
                    dest = 'device_type', default = 'cuda',
                    help = 'Enter number of epochs to use during training.')

parser.add_argument('--hidden_layer1', action = 'store',
                    dest = 'hidden_layer1', type = int, default = 25088,
                    help = 'Enter number of epochs to use during training.')

parser.add_argument('--hidden_layer2', action = 'store',
                    dest = 'hidden_layer2', type = int, default = 4096,
                    help = 'Enter number of epochs to use during training.')


parser.add_argument('--classes', action = 'store',
                    dest = 'num_classes', type = int, default = 102,
                    help = 'Enter number of epochs to use during training.')


results = parser.parse_args()
data_dir = results.data_directory
arch =  results.archt
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.units
epochs = results.num_epochs
hidden1 = results.hidden_layer1
hidden2 = results.hidden_layer2
nClasses = results.num_classes
device = results.device_type
#Define the processing unit and print basic enviroment info

device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
print("Processing unit:", device)
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print("If your are using a model diferent them vgg16 don't forget to tune the parameters")

model = utilidade.create_model(arch, hidden1, hidden2, dropout, nClasses)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
model.to(device)

trainloader, validloader, testloader, train_data = utilidade.load_data(data_dir)



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


if(utilidade.save_model(model, train_data, arch, dropout)):
    print('Training done, model saved')
else:
    print('Failed to save model')
    












