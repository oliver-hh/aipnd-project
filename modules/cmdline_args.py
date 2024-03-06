"""  
Command line argmuent parser
  
This module is designed to process the command line arguments for the two
scripts train.py and predict.py  
  
Author: Oliver Brandt  
Date: 2024-02-15
"""
import os
import argparse


def get_train_input_args():
    """  
    Parses command line arguments for training a model.  

    The function uses argparse to define and get command line arguments. The  
    following arguments are defined:  
    - data_dir (str): Path to the folder of data. This is a positional argument.  
    - save_dir (str, optional): Path to save the trained model. Default is '.'.  
    - checkpoint (str): Path to the checkpoint file that contains the trained network.
                        This is also a positional argument.  
    - arch (str, optional): CNN Model Architecture. Choices are
                            AlexNet, DenseNet121, VGG16. Default is 'DenseNet121'.
    - learning_rate (float, optional): Learning rate for model. Default is 0.01.  
    - hidden_units (int, optional): Number of hidden units in model. Default is 512.  
    - dropout_rate (float, optional): Dropout rate from 0 to 1. Default is 0.2.  
    - epochs (int, optional): Number of epochs for training. Default is 20.  
    - gpu (bool, optional): Use GPU for training if available. 

    Raises:
        ValueError: Data directory does not exist.
        ValueError: Learning rate is not a positive value.
        ValueError: Dropout rate is not between 0 and 1.
        ValueError: Hidden units is not a positive value.
        ValueError: Epochs is not a positive value.
        FileNotFoundError: Category names file does not exist.

    Returns:  
    Namespace: An argparse.Namespace instance with the arguments parsed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the folder of data'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='.',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--category_names',
        type=str,
        default='cat_to_name.json',
        help='File name for mapping of categories to real names'
    )
    parser.add_argument(
        '--arch',
        type=str,
        choices=['AlexNet', 'DenseNet121', 'VGG16'],
        default='DenseNet121',
        help='CNN Model Architecture'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Learning rate for model'
    )
    parser.add_argument(
        '--hidden_units',
        type=int,
        default=512,
        help='Number of hidden units in model'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.2,
        help='Dropout rate from 0 to 1'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for training if available'
    )

    args = parser.parse_args()

    # Custom error handling
    if not os.path.isdir(args.data_dir):
        raise ValueError(f'Data directory does not exist: {args.data_dir}')
    if not os.path.isdir(args.save_dir):
        raise ValueError(f'Save directory for checkpoint does not exist: {args.data_dir}')
    if args.learning_rate <= 0:
        raise ValueError('Learning rate must be positive.')
    if not 0 <= args.dropout_rate <= 1:
        raise ValueError('Dropout rate must be between 0 and 1.')
    if args.hidden_units <= 0:
        raise ValueError('Number of hidden units must be positive.')
    if args.epochs <= 0:
        raise ValueError('Number of epochs must be positive.')
    if not os.path.exists(args.category_names):
        raise FileNotFoundError(f'File {args.category_names} does not exist.')

    return args


def get_predict_input_args():
    """  
    Parses command line arguments for predicting the class of an image.  

    The function uses argparse to define and get command line arguments. The  
    following arguments are defined:  
    - input (str): Path to the image to classify. This is a positional argument.  
    - checkpoint (str): Path to the checkpoint file that contains the trained network.
                        This is also a positional argument.  
    - top_k (int, optional): Return top K most likely classes of the prediction. Default is 3.  
    - category_names (str, optional): File name for mapping of categories to real names.
                                      Default is 'cat_to_name.json'.  
    - gpu (bool, optional): Use GPU for inference if available.  

    Raises:
    FileNotFoundError: Input file does not exist.
    FileNotFoundError: Checkpoint file does not exist.
    FileNotFoundError: Category names file does not exist.

    Returns:  
    Namespace: An argparse.Namespace instance with the arguments parsed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input',
        type=str,
        help='Path to the image to classify'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to the checkpoint file that contains the trained network'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Return top K most likely classes of the prediction'
    )
    parser.add_argument(
        '--category_names',
        type=str,
        default='cat_to_name.json',
        help='File name for mapping of categories to real names'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for inference if available'
    )

    args = parser.parse_args()

    # Custom error handling
    if args.top_k < 1:
        raise ValueError('Top k must be greater or equal 1.')
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'File {args.input} does not exist.')
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'File {args.checkpoint} does not exist.')
    if not os.path.exists(args.category_names):
        raise FileNotFoundError(f'File {args.category_names} does not exist.')

    return args
