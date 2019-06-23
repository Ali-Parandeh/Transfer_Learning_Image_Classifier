# PROGRAMMER: Alireza Parandeh
# DATE CREATED: 22.06.2019                            
# REVISED DATE: 
# PURPOSE: Create utility functions for loading data and preprocessing images.

# Import here
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from torchvision import datasets, transforms, models
from args import get_prediction_input_args, get_training_input_args

# TODO: Define your transforms for the training, validation, and testing sets
def data_loader(data_dir):
    # I'm just going to put my hands in the training flower basket and shuffle, cut and skew my flowers
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    
    # TODO: Load the datasets with ImageFolder
    data = {}
    data["train_data"] = datasets.ImageFolder(train_dir, transform= train_transforms)
    data["test_data"] = datasets.ImageFolder(test_dir, transform= test_valid_transforms)
    data["valid_data"] = datasets.ImageFolder(valid_dir, transform= test_valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    data_loader = {}
    data_loader["train_loader"] = torch.utils.data.DataLoader(data["train_data"], 
                                                              batch_size=64, shuffle=True)
    data_loader["test_loader"]  = torch.utils.data.DataLoader(data["test_data"], 
                                                              batch_size=64, shuffle=True)
    data_loader["valid_loader"]  = torch.utils.data.DataLoader(data["valid_data"], 
                                                               batch_size=64, shuffle=True)
    return data, data_loader


def load_labels():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name
