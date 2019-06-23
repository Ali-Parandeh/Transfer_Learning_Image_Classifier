# PROGRAMMER: Alireza Parandeh
# DATE CREATED: 22.06.2019                            
# REVISED DATE: 
# PURPOSE: Create functions to train a new network on a dataset and save the model as a checkpoint. 
#          Also, Prints out training loss, validation loss, and validation accuracy as the network trains

import torch
from torchvision import datasets, transforms, models
from helper import data_loader, load_labels
from classifier import Classifier, Arguments
from args import get_training_input_args



def main():

    training_user_input = get_training_input_args()
    model_name = training_user_input.arch
    hidden_units = training_user_input.hidden_units
    dropout = training_user_input.dropout
    device = training_user_input.gpu
    learning_rate = training_user_input.learning_rate

    # data, data_loader =  data_loader(train_user_input.dir)
    
    
    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
        checkpoint_name = "vgg16_checkpoint.pth"
    else:
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
        checkpoint_name = "densenet121_checkpoint.pth"

    for param in model.parameters():
        param.requires_grad= False
        
    # Replacing the pre-trained model classifier with my own. The previous classifier parameters were frozen. Mine aren't so now if I train the network, only parameters of my classifier will be updated not the features layer.
    model.classifier = Classifier(input_features, 102, hidden_units, drop_p=dropout)
    hyperparams = Arguments(model, device, learning_rate)

    model.to(hyperparams.device)

    print(hyperparams.device)
    print(model)

if __name__ == "__main__":
    main()