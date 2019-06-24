# Transfer Learning Image Classifier

## Description

A command line application that:

- Lets users define their own classifier and hyperparameters, attach it to a Convolutional Neural Network, train it and save it to a checkpoint on any user-defined directory. 
- The users can also load other trained models (checkpoints) to classify images of from large number of classes and see top K classes & predictions. 
- Both training and prediction functions can be done on both the CPU or GPU as specified by the user.
- I have also provided a pre-trained CNN that can classify images of flowers up to 102 categories up to 93.5% accuracy.

In this project I first built this image classifier with PyTorch in Jupyter Notebooks, then converted it into a command line application.

**This is the final project that needs to be completed and submitted to Udacity for review as major part of passing their AI Programming with Python Nanodegree.**

For more information about this project and Udacity Nanodegree check this link: [Udacity AI Programming with Python Nanodegree](https://eu.udacity.com/course/ai-programming-python-nanodegree--nd089)

## Installation

First clone this repository using:
`https://github.com/Ali-Parandeh/Transfer_Learning_Image_Classifier.git`

Then, to configure the correct environment, use **requirements.txt** or the **environment.yml** files:

- Install from **requirements.txt** file:
`conda create --name <env_name> --file <.txt file>`

- Duplicate conda environment using **environment.yml** file:
`conda create --name <clone_name> --clone <env_name>`

You also need a folder structure in the following format to hold your image files:

`Image_folder/data_set/class_name/file_name.jpg`

```N/A
flowers/test/10/xxx.png
flowers/train/1/xxy.jpeg
flowers/valid/3/xxz.png
.
.
.
flowers/test/102/123.jpg
flowers/train/2/nsdf3.png
flowers/valid/1/asd932_.png
```

For more information, see Pytorch's tutorial on how to load data: [Pytorch's Data Loading and Processing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

## Usage

This repository contains two main application entry files `train.py` and `predict.py`:

- The first file, `train.py`, will train a new network on a dataset and save the model as a checkpoint.
- The second file, `predict.py`, uses a trained network to predict the class for an input image.

To train a new network on a data set with `train.py`:

**Basic usage:** `python train.py data_directory`

- Prints out training loss, validation loss, and validation accuracy as the network trains

_Extra Options:_

- Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
- Choose architecture: `python train.py data_dir --arch "vgg13"` 

> **NOTE: Only `densenet121` and `vgg16` pre-trained CNNs are currently supported in this application.**

- Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
- Use GPU for training: `python train.py data_dir --gpu`
- Predict image name from an image with `predict.py` along with the probability of that name. That is, the user can pass in a single image `/path/to/image` and return the image name and class probability.

**Basic usage:** `python predict.py /path/to/image checkpoint`

_Extra Options:_

- Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`
- Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
- Use GPU for inference: `python predict.py input checkpoint --gpu`
