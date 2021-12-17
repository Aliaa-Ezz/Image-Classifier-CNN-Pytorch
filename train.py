import argparse
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch import optim
from os.path import isdir
from collections import OrderedDict
import pandas as pd
import numpy as np
import json

#first get the parsing of the arguments done by specifying some arguments.
def parse():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--hidden_units',type=int)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    return args

#now we want to transform the images to be ready in training and predicting
def train_transformer(directory):
    train_ = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(directory, transform=train_)
    return train_data

def test_transformer(directory):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(directory, transform=test_transforms)
    return test_data

def valid_transformer(directory):
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(directory, transform=valid_transforms)
    return valid_data
#load the data using dataloader, choose your batchsize  
def load_data(data_loaded, trained=True):
    if trained: 
        load = torch.utils.data.DataLoader(data_loaded, batch_size=50, shuffle=True)
    else: 
        load = torch.utils.data.DataLoader(data_loaded, batch_size=50)
    return load
#check for the gpu wether its used or not
def gpu_used(gpu):
    if not gpu:
        return torch.device("cpu")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        print("CPU to go")
    return dev
#now to  load vgg
def model_vgg_load(arch="vgg16"):
    if type(arch) == type(None): 
        m = models.vgg16(pretrained=True)
        m.name = "vgg16"
        print("vgg16")
    else: 
        exec("m = models.{}(pretrained=True)".format(arch))
        m.name = arch
    
    for parameter in m.parameters():
        parameter.requires_grad = False 
    return m
#set up you classifier with the layer you want.
def classifier_(model, hidden):
    if type(hidden) == type(None): 
        hidden = 4096
        print("Number of layers is 4096.")

    input_features = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([
    ('x', nn.Linear(25088,1024)),
    ('relu1',nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('hidden1', nn.Linear(1024, 750)),
    ('relu2', nn.ReLU()),
    ('hidden2', nn.Linear(750,500)),
    ('relu3', nn.ReLU()),
    ('hidden3', nn.Linear(500,300)),
    ('relu4', nn.ReLU()),
    ('hidden4', nn.Linear(300,102)),
    ('y', nn.LogSoftmax(dim=1))
     
]))
    return classifier
#validate your model
def validate(model, tstload, how, dev):
    tstloss = 0
    acc = 0
    
    for ii, (ipts, lbls) in enumerate(tstload):
        
        ipts, lbls = ipts.to(dev), lbls.to(dev)
        
        output = model.forward(ipts)
        tstloss += how(output, lbls).item()
        
        ps = torch.exp(output)
        e = (lbls.data == ps.max(dim=1)[1])
        acc += e.type(torch.FloatTensor).mean()
    return tstloss, acc
#time for training the model
def train_model(model, trload,vlload, tstload, dev, 
                  how, optimizer, epochs, every, steps):
    if type(epochs) == type(None):
        epochs = 8
        print("Number of Epochs: {}".format(epochs))    
 
    print("Training....\n")


    for e in range(epochs):
        runloss = 0
        model.train()
        
        for ii, (ipts, lbls) in enumerate(trload):
            steps += 1
            
            ipts, lbls = ipts.to(dev), lbls.to(dev)
            
            optimizer.zero_grad()
            outputs = model.forward(ipts)
            loss = how(outputs, lbls)
            loss.backward()
            optimizer.step()
        
            runloss += loss.item()
        
            if steps % every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, acc = validate(model, vlload, how, 'cuda:0')
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(runloss/every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(tstload)),
                     "Accuracy of Validation: {:.4f}".format(acc/len(tstload)))
            
                runloss = 0
                model.train()

    return model
#validate your trained model
def validate_model(Model, tstload, dev):

    c = 0
    t = 0
    with torch.no_grad():
        Model.eval()
        for data in tstload:
            images, lbls = data
            images, lbls = images.to(dev), lbls.to(dev)
            outputs = Model(images)
            _, pred = torch.max(outputs.data, 1)
            t += lbls.size(0)
            c += (pred == lbls).sum().item()
    
    print('Accuracy on test data: %d%%' % (100 * c / t))

def save_checkpoint(Model, Save_Dir, data_training):

            Model.class_to_idx = data_training.class_to_idx
            

            checkpoint = {'arch': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}

            torch.save(checkpoint, 'checkpoint.pth')

def main():
 # now in the main program we will call the functions to process
#first the arg functions
    args = parse()
#specifying the directories we will work on
    flowers_direc = 'flowers'
    trainDir = flowers_direc + '/train'
    validDir = flowers_direc + '/valid'
    tstDir = flowers_direc + '/test'
    
#using the tranformation on the datasets
    train_ = test_transformer(trainDir)
    valid_ = train_transformer(validDir)
    test_ = train_transformer(tstDir)
 #loading the data   
    trload = load_data(train_)
    vlload = load_data(valid_, trained=False)
    tstload = load_data(test_, trained=False)
    
#load the vgg model
    model = model_vgg_load(arch=args.arch)
    
# use our classifier
    model.classifier = classifier_(model, hidden=args.hidden_units)
#determine the device used
    dev = gpu_used(gpu=args.gpu);

    model.to(dev);
#learning rate ceck
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
# what is the criterion used in the model
    how = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    every = 30
    steps = 0
# and now to train the model
    trained_model = train_model(model, trload, vlload, tstload, dev, how, optimizer, args.epochs, every, steps)
    
    print("\nTraining is Complete")
    validate_model(trained_model, tstload, dev)
    save_checkpoint(trained_model, 'checkpoint.pth', train_)

if __name__ == '__main__': main()