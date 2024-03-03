"""_summary_
Module flower_classifier classifies flowers by a freshly trained network or 
a loaded network. It containts the following classes and methods:
- Class FlowerClassifier
"""

from PIL import Image
import numpy as np

import torch
from torch import optim
from torchvision import models

# Constants
DEBUG = True
ALEXNET = 'AlexNet'
DENSENET121 = 'DenseNet121'
RESNET18 = 'ResNet18'
VGG16 = 'VGG16'

# Class FlowerClassifier


class FlowerClassifier:
    """_summary_
    Flower Classifier
    """

    def __init__(self,
                 # Common parameters
                 category_mapping=None, top_k=None, gpu=None,
                 # From checkpoint parameters
                 checkpoint_path=None,
                 # From training
                 training_path=None, arch=None, learning_rate=None, hidden_units=None,
                 epochs=None, dropout_rate=None):

        # Initializae common parameters
        self.category_mapping = category_mapping
        self.top_k = top_k

        # Set device for training/infernence (fall back on cpu if necessary)
        self.device = 'cuda:0' if gpu and torch.cuda.is_available() else 'cpu'

        # Initialize specific parameters
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
            self.model, self.optimizer = self.load_model_from_checkpoint()
        if DEBUG:
            print(f'Model loaded from checkpoint {self.checkpoint_path}')
            print(f'Architecture : {self.arch:>10}')
            print(f'Epochs       : {self.epochs:>10}')
            print(f'Learning rate: {self.learning_rate:>10}')
            print(f'Hidden units : {self.hidden_units:>10}')
            print(f'Dropout rate : {self.dropout_rate:>10}')

        elif (training_path is not None and arch is not None and learning_rate is not None and
              hidden_units is not None and epochs is not None):
            self.training_path = training_path
            self.arch = arch
            self.learning_rate = learning_rate
            self.hidden_units = hidden_units
            self.epochs = epochs
            self.dropout_rate = dropout_rate
            # Train the model
        else:
            raise ValueError("Incorrect constructor parameters")

    @classmethod
    def from_checkpoint(cls, checkpoint_path, category_mapping, top_k, gpu):
        """_summary_
        Generates an instance based on a checkpoint file. 

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            category_mapping (str): Path to the category mapping file.
            top_k (int): Nummber of the top categories of the prediction to be showwn.
            gpu (bool): Use GPU for inference if available.

        Returns:
            FlowerClassifier: Instance based on checkoint
        """
        return cls(checkpoint_path=checkpoint_path,
                   category_mapping=category_mapping, top_k=top_k, gpu=gpu)

    @classmethod
    def from_training_data(cls, training_path, arch, learning_rate, hidden_units, epochs,
                           category_mapping, top_k, gpu):
        """_summary_
        Generated an instance based on training data

        Args:
            training_path (str): Path to the training image with subdirectories test, train, valid.
            arch (string): Architecture of the model
            learning_rate (float): Learning rate
            hidden_units (int): Number of hideen units
            epochs (int): Number of epochs used to train the network
            category_mapping (string): Path to to file with the category mappings.
            top_k (int): Nummber of the top categories of the prediction to be showwn.
            gpu (bool): Use GPU for inference if available.

        Returns:
            FlowerClassifier: Instance based on checkoint
        """
        return cls(training_path=training_path, arch=arch, learning_rate=learning_rate,
                   hidden_units=hidden_units, epochs=epochs,
                   category_mapping=category_mapping, top_k=top_k, gpu=gpu)

    def get_pretrained_model(self, arch):
        """_summary_
        Returns a pre-trained model based on the architecture specified.
        Args:
            arch (str): Name of the architecture

        Raises:
            ValueError: If 'arch' does not match any of the specified architectures.
                        (AlexNet, DenseNet121, ResNet18, VGG16)

        Returns:
             torch.nn.Module: A pre-trained model of the specified architecture.  
        """
        if arch == ALEXNET:
            return models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        elif arch == DENSENET121:
            return models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        elif arch == RESNET18:
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif arch == VGG16:
            return models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            raise ValueError(f"Invalid arch value {arch}")

    def load_model_from_checkpoint(self):
        """_summary_
        Load the model from a file
        """
        # Load model data from file
        checkpoint_data = torch.load(
            self.checkpoint_path, map_location=torch.device(self.device))

        # Get metadata from loaded model
        self.arch = checkpoint_data['arch']
        self.epochs = checkpoint_data['epochs']
        self.learning_rate = checkpoint_data['learning_rate']
        self.hidden_units = checkpoint_data['hidden_units']
        self.dropout_rate = checkpoint_data['dropout_rate']

        # Instantiate model from pre-trained model and set classifier
        model = self.get_pretrained_model(self.arch)
        model.classifier = checkpoint_data['classifier']

        # Restore model data
        model.load_state_dict(checkpoint_data['model_state_dict'])
        model.class_to_idx = checkpoint_data['class_to_idx']

        # Restore optimizer
        optimizer = optim.Adam(
            model.classifier.parameters(), lr=self.learning_rate)
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

        return model, optimizer


    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        # Load image and resize it to the shortest side
        pil_image = Image.open(image)
        width, height = pil_image.size
        if height < width:
            new_height = 256
            new_width = int(256 * width / height)
        else:
            new_width = 256
            new_height = int(256 * height / width)
        pil_image = pil_image.resize((new_width, new_height))

        # Crop from the center of the imamge
        margin = (256 - 224) / 2
        (left, upper, right, lower) = (margin, margin, 256-margin, 256-margin)
        pil_image = pil_image.crop((left, upper, right, lower))

        # Convert image to numpy array and scale values between 0 and 1
        np_image = np.array(pil_image) / 255

        # Normalize the image  and transpose it so that
        # the channel information is the first dimension.
        means_per_channel = [0.485, 0.456, 0.406]
        stddevs_per_channel = [0.229, 0.224, 0.225]
        np_image = (np_image - means_per_channel) / stddevs_per_channel
        np_image = np_image.transpose((2, 0, 1))
        return np_image

    def classify_image(self, image_path):
        """_summary_
        Classify image
        """
        # Image processing (resize, crop and normalize)
        numpy_image = self.process_image(image_path)

        # Convert image to a tensor
        image = torch.from_numpy(numpy_image)
        # Add another dimension (required for batch_processing)
        image = image.unsqueeze(0).float()

        # Set model to evaluation mode and determine the outputs for the image
        self.model.eval()
        with torch.no_grad():
            self.model, image = self.model.to(self.device), image.to(self.device)
            log_outputs = self.model.forward(image)
            outputs = torch.exp(log_outputs)

        # Get the top 5 probabilities for matching classes
        top_k, top_indices = outputs.topk(5, dim=1)

        # Convert/flatten top_k and top_indices from a tensor to an one-dimensional array
        top_k = top_k.numpy().flatten().tolist()
        top_indices = top_indices.numpy().flatten()

        # Invert class_to_idx and map indices to the actual class labels
        idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}
        top_classes = [idx_to_class[top_indices[i]] for i in range(5)]

        return top_k, top_classes


if __name__ == "__main__":
    checkpoint_classifier = FlowerClassifier.from_checkpoint(
        './checkpoint.pth', category_mapping='./cat_to_name.json', top_k=5, gpu=True)

    probabilities, classes = checkpoint_classifier.classify_image(
        './flowers/test/10/image_07090.jpg')

    print(probabilities)
    print(classes)

    # trained_classifier = FlowerClassifier.from_training_data(
    #     "path/to/training/data", "vgg16", 0.01, 100, 10,
    #     category_mapping="path/to/mapping", top_k=5, gpu=True)
