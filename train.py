# PROGRAMMER: Alireza Parandeh
# DATE CREATED: 22.06.2019                            
# REVISED DATE: 
# PURPOSE: Create functions to train a new network on a dataset and save the model as a checkpoint. 
#          Also, Prints out training loss, validation loss, and validation accuracy as the network trains

import torch
from torchvision import datasets, transforms, models
from helper import data_loader, 
from classifier import Classifier, Arguments
from args import get_training_input_args

training_user_input = get_training_input_args()

def main():  


    model = models.densenet121(pretrained= True)

    for param in model.parameters():
        param.requires_grad= False
        
    # Replacing the pre-trained model classifier with my own. The previous classifier parameters were frozen. Mine aren't so now if I train the network, only parameters of my classifier will be updated not the features layer.
    model.classifier = Classifier(input_units, 102, training_user_input.hidden_units)
    model.to(device)

    hyperparams = Arguments()

    print(device)
    print(model)

if __name__ == "__main__":
    main()