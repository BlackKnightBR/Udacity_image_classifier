import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

from util_functions import load_data, process_image
from functions import load_checkpoint, predict, test_model


parser = argparse.ArgumentParser(description='Use neural network to make prediction on image.')

parser.add_argument('--image_path', action='store',
                    default = '../aipnd-project/flowers/test/102/image_08004',
                    help='Enter path to image.')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default = 'checkpoint.pth',
                    help='Enter location to save checkpoint in.')

parser.add_argument('--arch', action='store',
                    dest='pretrained_model', default='vgg11',
                    help='Enter pretrained model to use, default is VGG-11.')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 3,
                    help='Enter number of top most likely classes to view, default is 3.')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='Enter path to image.')

parser.add_argument('--gpu', action="store_true", default=False,
                    help='Turn GPU mode on or off, default is off.')

results = parser.parse_args()

save_dir = results.save_directory
image = results.image_path
top_k = results.topk
gpu_mode = results.gpu
cat_names = results.cat_name_dir
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

# Establish model template
pre_tr_model = results.pretrained_model
model = getattr(models,pre_tr_model)(pretrained=True)

# Load model
loaded_model = load_checkpoint(model, save_dir, gpu_mode)

# Preprocess image - assumes jpeg format
processed_image = process_image(image)

if gpu_mode == True:
    processed_image = processed_image.to('cuda')
else:
    pass

# Carry out prediction
probs, classes = predict(processed_image, loaded_model, top_k, gpu_mode)

# Print probabilities and predicted classes
print(probs)
print(classes)

names = []
for i in classes:
    names += [cat_to_name[i]]

# Print name of predicted flower with highest probability
print(f"This flower is most likely to be a: '{names[0]}' with a probability of {round(probs[0]*100,4)}% ")

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





# Function to load and preprocess test image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Converting image to PIL image using image file path
    pil_im = Image.open(f'{image}' + '.jpg')

    # Building image transform
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    ## Transforming image for use with network
    pil_tfd = transform(pil_im)

    # Converting to Numpy array
    array_im_tfd = np.array(pil_tfd)

    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(array_im_tfd).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    return img_add_dim

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


def load_checkpoint(model, save_dir, gpu_mode):


    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def predict(processed_image, loaded_model, topk, gpu_mode):
    # Predict the class (or classes) of an image using a trained deep learning model.

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()

    if gpu_mode == True:
        loaded_model.to('cuda')
    else:
        loaded_model.cpu()

    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(processed_image)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]

    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])

    # Loading index and class mapping
    class_to_idx = loaded_model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]

    return probs_top_list, classes_top_list
