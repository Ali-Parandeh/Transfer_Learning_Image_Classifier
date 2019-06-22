# PROGRAMMER: Alireza Parandeh
# DATE CREATED: 22.06.2019                            
# REVISED DATE: 
# PURPOSE:  Create two functions that retrieve the following command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. 

#       train.py
#           Basic usage: python train.py data_directory
#           
#           Options:
#               Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#               Choose architecture: python train.py data_dir --arch "vgg13"
#               Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#               Use GPU for training: python train.py data_dir --gpu
 
#       predict.py   
#           Basic usage: python predict.py /path/to/image checkpoint
#           Options:
#               Return top KK most likely classes: python predict.py input checkpoint --top_k 3
#               Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#               Use GPU for inference: python predict.py input checkpoint --gpu


import argparse
import torch

def get_training_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. The following command line 
    arguments let the user specify the model training specification.
    Command Line Arguments:
      1. Image Folder as --dir with default value '/flowers'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Hyperparameter --learning_rate with default value 0.001
      4. Hyperparameter --hidden_units with default value 1000
      5. Hyperparameter --epochs with default value 10
      6. Hyperparameter --dropout with default value of 0.2
      7. Device --gpu with default value of 'cpu'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object 
    """
  
    parser = argparse.ArgumentParser(description= "Process the command line arguments provide by the user")
    parser.add_argument("--dir", type=str, help="Path to the folder of flower images" )
    parser.add_argument("--arch", type=str, default="densenet121", help="Type of CNN model architecture to use")
    parser.add_argument("--learning_rate", type=float, default= 0.001, help="The learning rate of the CNN training")
    parser.add_argument("--hidden_units", type=int, default= 1000, help="Number of units in the hidden layer of the classifier")
    parser.add_argument("--epochs", type=int, default= 10, help="Number of epochs for training")
    parser.add_argument("--dropout", type=float, default= 0.2, help="Probability of unit dropout")
    parser.add_argument("--gpu",  action='store_true', help="Select to run the programme with GPU")

    return parser.parse_args()

def get_prediction_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. The following command line 
    arguments let the user specify specifications for predictions.
    
    Command Line Arguments:
      1. Image Folder as --dir with default value '/flowers'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Hyperparameter --learning_rate with default value 0.001
      4. Hyperparameter --hidden_units with default value 1000
      5. Hyperparameter --epochs with default value 10
      6. Device --gpu with default value of 'None', True if --gpu is specified in the command line
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object 
    """
  
    parser = argparse.ArgumentParser(description= "Process the command line arguments provide by the user")
    parser.add_argument("input", type=str, required= True, help="Path to the folder of flower images" )
    parser.add_argument("checkpoint", type=str, required= True, help="Type of CNN model architecture to use")
    parser.add_argument("--top_k", type=int, default= 3, help="The learning rate of the CNN training")
    parser.add_argument("--category_names", type=str, help="Number of units in the hidden layer of the classifier")
    parser.add_argument("--gpu", action='store_true', help="Select to run the programme with GPU")

    return parser.parse_args()


def main():
      x = get_training_input_args()
      device = torch.device("cuda:0" if x.gpu else "cpu")
      print(device)

if __name__ == "__main__":
    main()