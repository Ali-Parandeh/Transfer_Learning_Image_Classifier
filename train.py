# PROGRAMMER: Alireza Parandeh
# DATE CREATED: 22.06.2019
# REVISED DATE:
# PURPOSE: Create functions to train a new network on a dataset and save the model as a checkpoint.
#          Also, Prints out training loss, validation loss, and validation accuracy as the network trains

import torch
from torch import nn
from torchvision import datasets, transforms, models
from classifier import Classifier, Arguments
from args import get_training_input_args


# Defining transforms for the training, validation, and testing sets
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

    # Loading the datasets with ImageFolder
    data = {}
    data["train_data"] = datasets.ImageFolder(train_dir, transform=train_transforms)
    data["test_data"] = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    data["valid_data"] = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    data_iterator = {}
    data_iterator["train_loader"] = torch.utils.data.DataLoader(data["train_data"],
                                                                batch_size=64, shuffle=True)
    data_iterator["test_loader"] = torch.utils.data.DataLoader(data["test_data"],
                                                               batch_size=64, shuffle=True)
    data_iterator["valid_loader"] = torch.utils.data.DataLoader(data["valid_data"],
                                                                batch_size=64, shuffle=True)
    return data, data_iterator


# Trains the Network
def train(model, device, train_iterator, valid_iterator, criterion, optimizer, epochs):
    ''' Trains a neural network.

        Arguments
        ---------
        model: Un-trained Model
        train_iterator: a data generator that can provide a batch of data for training the model
        valid_iterator: a data generator that can provide a batch of data for validating the model
        criterion: Torch Function, Is used to calculate the error loss
        device: string that specifies what hardware the validation should be run on
        optimizer: Torch Module, Optimises the model weights per epoch
        
        Outputs
        ---------
        model: Trained Model
        training_losses: A list of average training losses in percentage of the batch per epoch
        validation_losses: A list of average validation losses of the batch per epoch
        validation_accuracies: A list of average accuracies in percentage of the batch per epoch

    '''
    for e in range(epochs):
        training_loss = 0
        validation_loss = 0
        validation_accuracy = 0
        model.train()
        # Perform a training pass for all images
        for images, labels in train_iterator:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            train_logits = model(images)
            loss = criterion(train_logits, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        # Perform a validation pass for all images
        else:

            with torch.no_grad():
                validation_loss, validation_accuracy = validation(model, valid_iterator, criterion, device)

                print(f"Device = {device}", "Epoch {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(
                          training_loss/len(train_iterator)),
                      "Validation Loss: {:.3f}.. ".format(
                          validation_loss/len(valid_iterator)),
                      "Validation Accuracy: {:.3f}".format(validation_accuracy/len(valid_iterator)))

                model.train()

    return model


def validation(model, data_iterator, criterion, device):
    ''' Validates the predictions of the network.

        Arguments
        ---------
        - model: dictionary, neural network model
        - data_iterator: a data iterator that can provide a batch of data for validation
        - criterion: Torch function, Is used to calculate the error loss
        - device: string that specifies what hardware the validation should be run on
        
        Outputs
        ---------
        - test_loss: Average loss per batch
        - accuracy: Average accuracy in percentage per batch

    '''
    model.eval()
    validation_loss = 0
    validation_accuracy = 0
    for images, labels in data_iterator:
        images, labels = images.to(device), labels.to(device)
        valid_logits = model.forward(images)
        validation_loss += criterion(valid_logits, labels)
        ps = torch.exp(valid_logits)
        top_p, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        validation_accuracy += torch.mean(equality.type(torch.FloatTensor))

    return validation_loss, validation_accuracy


# Save the checkpoint
def save_model(model, data, checkpoint_dir, optimizer):
    ''' Saves the model checkpoint into a tph file
    '''
    model.class_to_idx = data.class_to_idx

    checkpoint = {"class_to_idx": data.class_to_idx,
                  "model": model,
                  "state_dict": model.state_dict(),
                  "optimiser_state_dict": optimizer.state_dict()}

    torch.save(checkpoint, checkpoint_dir)

    print("Model Saved!")


def main():
    # Get user inputs for constructing and training a model.
    ti = get_training_input_args()
    data, data_iterator = data_loader(ti.dir)

    if ti.arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
        checkpoint_dir = ti.save_dir + "vgg16_checkpoint.pth"
    else:
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
        checkpoint_dir = ti.save_dir + "densenet121_checkpoint.pth"

    # Freeze all parameters. When the classifier later on is replaced with our own, only our classifier parameters will be updated during training.
    for param in model.parameters():
        param.requires_grad = False

    # Replacing the pre-trained model classifier with my own. The previous classifier parameters were frozen. Mine aren't so now if I train the network, only parameters of my classifier will be updated not the features layer.
    model.classifier = Classifier(input_features, 102, ti.hidden_units, drop_p=ti.dropout)
    hyperparams = Arguments(model, ti.gpu, ti.learning_rate, ti.epochs)
    model = model.to(hyperparams.device)

    model = train(
        model,
        hyperparams.device,
        data_iterator["train_loader"],
        data_iterator["train_loader"],
        hyperparams.criterion,
        hyperparams.optimizer,
        hyperparams.epochs)

    save_model(model, data["train_data"], checkpoint_dir, hyperparams.optimizer)


if __name__ == "__main__":
    main()
