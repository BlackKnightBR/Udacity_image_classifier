import argparse
import torchvision
import torch
import function as utilidades
import json


parser = argparse.ArgumentParser()

parser.add_argument('--image_path', action='store',
                    default = 'flowers/test/11/image_03098.jpg',
                    help='Enter path to image.')

parser.add_argument('--model', action='store',
                    dest='model_path', default= 'checkpoint.pth',
                    help='Enter location of trained model.')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 5,
                    help='Enter number of top most likely classes to view.')

parser.add_argument('--labels', action='store',
                    dest='labels',
                    default = 'cat_to_name.json',
                    help='Enter path to image labels.')

parser.add_argument('--device', action = 'store',
                    dest = 'device_type', default = 'cuda',
                    help = 'Enter number of epochs to use during training.')



results = parser.parse_args()

image = results.image_path
path = results.model_path
topK = results.topk
labels = results.labels
device = results.device_type

#Define the processing unit and print basic enviroment info
device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
print("Processing unit:", device)
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

with open(labels, 'r') as f:
    labels = json.load(f)


model = utilidades.load_model(path)
probability, classes = utilidades.predict(image, model, topK, labels, device)


# Print probabilities and predicted classes
for percent, classN in zip(probability,classes):
    percent *= 100
    print("Probality of flower class: {:.2f}%, Name of the class: {}".format(percent, classN))
