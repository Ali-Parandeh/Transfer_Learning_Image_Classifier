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
      4. Hyperparameter --hidden_units with default value 512
      5. Hyperparameter --epochs with default value 10
      6. Device --gpu with default value of 'cpu'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object 
    """
    parser.