
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


parser = argparse.ArgumentParser(description='Train neural network.')

# ../aipnd-project/flowers
parser.add_argument('data_directory', action = 'store',
                    help = 'Enter path to training data.')

parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg11',
                    help= 'Enter pretrained model to use; this classifier can currently work with\
                           VGG and Densenet architectures. The default is VGG-11.'

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.001,
                    help = 'Enter learning rate for training the model, default is 0.001.')

parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=int, default = 0.05,
                    help = 'Enter dropout for training the model, default is 0.05.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=int, default = 500,
                    help = 'Enter number of hidden units in classifier, default is 500.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 2,
                    help = 'Enter number of epochs to use during training, default is 1.')

parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off, default is off.')

results = parser.parse_args()

data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.units
epochs = results.num_epochs
gpu_mode = results.gpu

# Load and preprocess data
trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)

# Load pretrained model
pre_tr_model = results.pretrained_model
model = getattr(models,pre_tr_model)(pretrained=True)

# Build and attach new classifier
input_units = model.classifier[0].in_features
build_classifier(model, input_units, hidden_units, dropout)

# Recommended to use NLLLoss when using Softmax
criterion = nn.NLLLoss()
# Using Adam optimiser which makes use of momentum to avoid local minima
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# Train model
model, optimizer = train_model(model, epochs,trainloader, validloader, criterion, optimizer, gpu_mode)

# Test model
test_model(model, testloader, gpu_mode)
# Save model
save_model(loaded_model, train_data, optimizer, save_dir, epochs)

# Function to load and preprocess the data
def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # The validation set will use the same transform as the test set
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    # The trainloader will have shuffle=True so that the order of the images do not affect the model
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return trainloader, testloader, validloader, train_data, test_data, valid_data

# Function to build new classifier
def build_classifier(model, input_units, hidden_units, dropout):
    # Weights of pretrained model are frozen so we don't backprop through/update them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Replacing the pretrained classifier with the one above
    model.classifier = classifier
    return model



def validation(model, validloader, criterion, gpu_mode):
    valid_loss = 0
    accuracy = 0

    if gpu_mode == True:
    # change model to work with cuda
        model.to('cuda')
    else:
        pass
    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(validloader):

        if gpu_mode == True:
        # Change images and labels to work with cuda
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            pass
        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy



def train_model(model, epochs,trainloader, validloader, criterion, optimizer, gpu_mode):
    #epochs = 3
    steps = 0
    print_every = 10

    if gpu_mode == True:
    # change to cuda
        model.to('cuda')
    else:
        pass

    for e in range(epochs):
        #since = time.time()
        running_loss = 0

        # Carrying out training step
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            if gpu_mode == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                pass

            # zeroing parameter gradients
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Carrying out validation step
            if steps % print_every == 0:
                # setting model to evaluation mode during validation
                model.eval()
                # Gradients are turned off as no longer in training
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, gpu_mode)
                print(f"No. epochs: {e+1}, \
                Training Loss: {round(running_loss/print_every,3)} \
                Valid Loss: {round(valid_loss/len(validloader),3)} \
                Valid Accuracy: {round(float(accuracy/len(validloader)),3)}")

                running_loss = 0
                # Turning training back on
                model.train()

    return model, optimizer



def test_model(model, testloader, gpu_mode):
    correct = 0
    total = 0

    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloader):

            if gpu_mode == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test accuracy of model for {total} images: {round(100 * correct / total,3)}%")



def save_model(model, train_data, optimizer, save_dir, epochs):
    # Saving: feature weights, new classifier, index-to-class mapping, optimiser state, and No. of epochs
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs}

    return torch.save(checkpoint, save_dir)



def load_checkpoint(model, save_dir, gpu_mode):


    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model
