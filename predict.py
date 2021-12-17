import argparse
from PIL import Image
from math import ceil
from train import gpu_used
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch import optim
from os.path import isdir
from collections import OrderedDict
import pandas as pd
import numpy as np
import json


# to start prediction we first want to parse all the arguments that we defined before in the train part
def parse():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--top_k', type=int )
    parser.add_argument('--category', type=str)
    parser.add_argument('--gpu', action="store")
    args = parser.parse_args()  
    return args



def checkpoint_(checkpoint_path):
    checkpoint = torch.load("checkpoint.pth")   
#to retrive the data from the checkpoint
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['arch'])
        model.name = checkpoint['arch']    
#we want to freeze the params now
    for param in model.parameters(): param.requires_grad = False   
#time to load the rest of the checkpoints
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])   
    return model


def process_image(image):
    im = Image.open(image)
    tw, th = im.size
    
    if tw<th:
        size = [256, 256**600]
    else:
        size=[256**600, 256]
    im.thumbnail(size)
    c = tw/4, th/4
    left, top, r, b = c[0] - 122, c[1] - 122, c[0]+122, c[1]+122
    im = im.crop((left, top, r, b))
    nmp_im = np.array(im)/255
    
    mean= [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    nmp_im = (nmp_im - mean)/std
    nmp_im = nmp_im.transpose(2,0,1)
    
    return nmp_im


#this functin is to predict the image we adjusted based on the checkpoint model we trained before
def predict(image_tensor, model, dev, cat_to_name, top_k):
    #adjusting that we want the top 5 classes of predictions
    if type(top_k) == type(None):
        top_k = 5    
    model.eval();
    torch_image = torch.from_numpy(np.expand_dims(image_tensor,axis=0)).type(torch.FloatTensor)
    model=model.cpu()
    log_probs = model.forward(torch_image)
#convert to linear
    linear_probs = torch.exp(log_probs)
#getting the top 5 classes
    top_probs, top_labels = linear_probs.topk(top_k)
#detaching
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]    
#convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    return top_probs, top_labels, top_flowers



#time of truth
def print_probability(probs, flowers):    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "is: {}, about: {}%".format(j[1], ceil(j[0]*100)))

        
        
def main():
#run functions 
    args = parse()
    with open('cat_to_name.json', 'r') as f:
        	cat_to_name = json.load(f)
    model = checkpoint_(args.checkpoint)
    image_tensor = process_image(args.image)
    device = gpu_used(gpu=args.gpu);
    top_probs, top_labels, top_flowers = predict(image_tensor, model,device, cat_to_name,args.top_k)
    print_probability(top_flowers, top_probs)

if __name__ == '__main__': main()