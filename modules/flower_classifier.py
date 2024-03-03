"""_summary_
Module flower_classifier classifies flowers by a freshly trained network or 
a loaded network. It containts the following classes and methods:
- Class FlowerClassifier
"""
import os
import time
import json
from collections import OrderedDict
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import (
    datasets,
    transforms,
    models
)

# Constants
DEBUG = True
ALEXNET = 'AlexNet'
DENSENET121 = 'DenseNet121'
VGG16 = 'VGG16'
OUTPUT_SIZE = 102
BATCH_SIZE=128

# Class FlowerClassifier


class FlowerClassifier:
    """_summary_
    Flower Classifier
    """

    def __init__(self,
                 # Common parameters
                 data_dir= None, category_mapping=None, top_k=None, gpu=None,
                 # From checkpoint parameters
                 checkpoint_path=None,
                 # From training
                 save_dir=None, arch=None, learning_rate=None, hidden_units=None,
                 dropout_rate=None, epochs=None):

        # Initializae common parameters
        self.data_dir = data_dir
        self.category_mapping = category_mapping
        self.top_k = top_k

        # Set device for training/infernence (fall back on cpu if necessary)
        self.device = 'cuda:0' if gpu and torch.cuda.is_available() else 'cpu'

        # Initialize specific parameters
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
            self.model, self.optimizer = self.load_model_from_checkpoint()
            if DEBUG:
                print(f'\nModel loaded from checkpoint {self.checkpoint_path}')

        elif (data_dir is not None and save_dir is not None and arch is not None and
              learning_rate is not None and hidden_units is not None and epochs is not None):
            self.data_dir = data_dir
            self.save_dir = save_dir
            self.arch = arch
            self.learning_rate = learning_rate
            self.hidden_units = hidden_units
            self.epochs = epochs
            self.dropout_rate = dropout_rate

            self.data_loaders, self.image_datasets = self.create_dataloaders()

            self.model, self.optimizer = self.create_pretrained_model()
            self.model.to(self.device)

            if DEBUG:
                print('\nModel created from training')

        else:
            raise ValueError("Incorrect constructor parameters")

        if DEBUG:
            print(f'Architecture : {self.arch:>10}')
            print(f'Epochs       : {self.epochs:>10}')
            print(f'Learning rate: {self.learning_rate:>10}')
            print(f'Hidden units : {self.hidden_units:>10}')
            print(f'Dropout rate : {self.dropout_rate:>10}\n')

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
    def from_training_data(cls, data_dir, save_dir, arch, learning_rate, hidden_units,
                           dropout_rate, epochs, category_mapping, top_k, gpu):
        """_summary_
        Generated an instance based on training data

        Args:
            data_dir (str): Path to the training image with subdirectories test, train, valid.
            save_dir (str): Path to the directory where the checkpoints will be saved.
            arch (string): Architecture of the model
            learning_rate (float): Learning rate
            hidden_units (int): Number of hideen units
            dropout_rate (float): Dropout rate from 0 to 1
            epochs (int): Number of epochs used to train the network
            category_mapping (string): Path to to file with the category mappings.
            top_k (int): Nummber of the top categories of the prediction to be showwn.
            gpu (bool): Use GPU for inference if available.

        Returns:
            FlowerClassifier: Instance based on checkoint
        """
        return cls(data_dir, save_dir=save_dir, arch=arch, learning_rate=learning_rate,
                   hidden_units=hidden_units, dropout_rate=dropout_rate, epochs=epochs,
                   category_mapping=category_mapping, top_k=top_k, gpu=gpu)

    def get_pretrained_model(self, arch):
        """_summary_
        Returns a pre-trained model based on the architecture specified.
        Args:
            arch (str): Name of the architecture

        Raises:
            ValueError: If 'arch' does not match any of the specified architectures.
                        (AlexNet, DenseNet121, VGG16)

        Returns:
             torch.nn.Module: A pre-trained model of the specified architecture.  
        """
        if arch == ALEXNET:
            return models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        elif arch == DENSENET121:
            return models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        elif arch == VGG16:
            return models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            raise ValueError(f"Invalid arch value {arch}")

    def create_pretrained_model(self):
        """_summary_
        Configure pretrained model
        """
        # Get an instance of a pretrained model. e.g. VGG16, ResNet50, DenseNet121, etc.)
        model = self.get_pretrained_model(self.arch)

        # Determine layer-sizes of the neural network
        if self.arch == 'DenseNet121':
            inputs_size = model.classifier.in_features
        else:
            # AlexNet, VGG16
            first_linear_layer = next(
                layer for layer in model.classifier.children() if isinstance(layer, torch.nn.Linear)
            )  
            inputs_size = first_linear_layer.in_features

        output_size = OUTPUT_SIZE 
        hl1_size = self.hidden_units
        hl2_size = self.hidden_units // 2

        # Freeze parameters to prevent backpropagation
        for param in model.parameters():
            param.requires_grad = False

        # Customize the model to the current classification and
        # add dropout to prevent overfitting
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(inputs_size, hl1_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p = self.dropout_rate)),
            ('fc2', nn.Linear(hl1_size, hl2_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p = self.dropout_rate)),
            ('fc3', nn.Linear(hl2_size, output_size)),
            ('output', nn.LogSoftmax(dim = 1))
        ]))
        model.classifier = classifier

        # Map classes to indices and create optimizer
        model.class_to_idx = self.image_datasets['train'].class_to_idx  
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)

        return model, optimizer

    def create_dataloaders(self):
        """_summary_
        Create dataloaders
        """
        train_dir = self.data_dir + '/train'
        valid_dir = self.data_dir + '/valid'
        test_dir = self.data_dir + '/test'

        # Define your transforms for the training, validation, and testing sets
        random_rotation = 30
        image_size = 224
        means_per_channel = [0.485, 0.456, 0.406]
        stddevs_per_channel = [0.229, 0.224, 0.225]

        data_transforms = {
            'train':    transforms.Compose([
                            transforms.RandomRotation(random_rotation),
                            transforms.RandomResizedCrop(image_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(means_per_channel, stddevs_per_channel),
                        ]),
            'valid':    transforms.Compose([
                            transforms.Resize(image_size + 30),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(means_per_channel, stddevs_per_channel),
                        ]),
            'test':     transforms.Compose([
                            transforms.Resize(image_size + 30),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(means_per_channel, stddevs_per_channel),
                        ])
        }

        # Load the datasets with ImageFolder
        image_datasets = {
            'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
            'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
            'test':  datasets.ImageFolder(test_dir,  transform=data_transforms['test'])
        }

        # Using the image datasets and the trainforms, define the dataloaders
        data_loaders =  {
            'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
            'valid': DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE),
            'test':  DataLoader(image_datasets['test'],  batch_size=BATCH_SIZE)
        }

        return data_loaders, image_datasets

    def train_network(self):
        """_summary_
        Train the neural network
        """
        print(f'Start training for device {self.device}')

        # Initialize variables
        criterion = nn.NLLLoss()
        steps = 0
        running_train_loss = 0
        print_every = 10

        # Set model to training mode
        self.model.train()

        # Start measurement
        start_time = time.time()

        # Iterate over all epochs to train the network
        for epoch in range(1, self.epochs+1):
            # Get inputs and labels of the current batch
            train_loader = self.data_loaders['train']
            for train_inputs, train_labels in train_loader:
                steps += 1

                # Move inputs and labels to device
                train_inputs = train_inputs.to(self.device)
                train_labels = train_labels.to(self.device)

                # Forward pass through the neural network and compute the loss of the model
                train_log_outputs = self.model.forward(train_inputs)
                train_loss = criterion(train_log_outputs, train_labels)

                # Clear old gradients, backpropagate, update model params with computed gradient
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                # Add current loss to the running loss
                running_train_loss += train_loss.item()

                # Test the network after n batches and calulate train loss and test loss/accuracy
                if steps % print_every == 0:
                    # Initialize test loss and accuracy
                    running_test_loss = 0
                    accuracy = 0

                    # Set model to evaluation mode
                    self.model.eval()

                    # Determine train/test-loss and accuracy with test data withoud gradient
                    with torch.no_grad():
                        # Perform the steps for test similar to training (see above)
                        test_loader = self.data_loaders['test']
                        for test_inputs, test_labels in test_loader:
                            test_inputs = test_inputs.to(self.device)
                            test_labels = test_labels.to(self.device)
                            test_log_outputs = self.model.forward(test_inputs)
                            test_loss = criterion(test_log_outputs, test_labels)
                            running_test_loss += test_loss.item()

                            # Calculate accuracy of the network so far
                            test_output = torch.exp(test_log_outputs)
                            _, top_class = test_output.topk(1, dim=1)
                            equals = top_class == test_labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    # Output network metrics
                    print(f'Epoch {epoch}/{self.epochs}.. '
                        f'Train loss: {running_train_loss/print_every:.3f}.. '
                        f'Test loss: {running_test_loss/len(test_loader):.3f}.. '
                        f'Test accuracy: {accuracy/len(test_loader):.3f}.. '
                        f'Time elapsed: {int(time.time()-start_time)}s')

                    # Reset model to training mode and reset running_train_loss for next batch
                    self.model.train()
                    running_train_loss = 0

        end_time = time.time()
        elapsed_time = int(end_time - start_time)

        print(f'End training for device {self.device}, duration={elapsed_time}s')

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

    def save_checkpoint(self):
        """_summary_
        Save checkpoint
        """

        # Collect model data which should be saved
        checkpoint_data = {
            'arch': 'DenseNet121',
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'model_state_dict': self.model.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict(),
            'classifier': self.model.classifier,
            'class_to_idx': self.model.class_to_idx
        }

        # Create output folder for checkpoints if necessary
        if not os.path.exists(self.save_dir):  
            os.makedirs(self.save_dir)  

        # Save model data to file checkpoint.pth
        torch.save(checkpoint_data, os.path.join(self.save_dir, 
                   f'checkpoint_{self.arch}_{self.epochs}_{self.hidden_units}_'
                   f'{self.learning_rate}_{self.dropout_rate}.pth'))

    def get_modelaccuracy(self):
        """_summary_
        Determine the accuracy of the model with data it has not seen yet
        """
        # Initialize variables, assign model to device and set it to evaluation mode
        total_images = 0
        correct_image_matches = 0
        self.model.eval()

        # Get validation data loader
        valid_loader = self.data_loaders['valid']
        # Disable gradient computations, which are not needed during model evaluation
        with torch.no_grad():
            # Iterate over all validation that is unknown (not trained/tested yet)
            for valid_images, valid_labels in valid_loader:
                valid_images = valid_images.to(self.device)
                valid_labels = valid_labels.to(self.device)
                valid_outputs = self.model(valid_images)

                # Get class with highest probs, compare with the labels / count correct matches
                _, predicted_class = torch.max(valid_outputs.data, 1)
                total_images += valid_labels.size(0)
                correct_image_matches += (predicted_class == valid_labels).sum().item()

        # Calculate accuracy
        accuracy = correct_image_matches / total_images
        return accuracy 

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

    def classify_image(self, image_path, category_names):
        """_summary_
        Classify image
        """
        # Load categories in dictionary
        with open(category_names, 'r', encoding='utf-8') as f:
            cat_to_name = json.load(f)

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
        top_k = top_k.cpu().numpy().flatten().tolist()
        top_indices = top_indices.cpu().numpy().flatten()

        # Invert class_to_idx and map indices to the actual class labels
        idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}
        top_classes = [idx_to_class[top_indices[i]] for i in range(5)]

        # Map category names
        category_class_mapping = [{
            'flower': cat_to_name[cls],
            'probability': round(prob, 3)
        } for cls, prob in zip(top_classes, top_k)]

        return category_class_mapping

if __name__ == "__main__":
    USECASE_CHECKPOINT = 'checkpoint'
    USECASE_TRAINING = 'training'
    USECASE = USECASE_TRAINING

    IMAGE_PATH = './flowers/test/10/image_07090.jpg'
    CAT_NAMES = './cat_to_name.json'


    if USECASE == USECASE_CHECKPOINT:
        # Test classification with saved checkpoint file
        checkpoint_classifier = FlowerClassifier.from_checkpoint(
            './checkpoint.pth', category_mapping='./cat_to_name.json', top_k=5, gpu=True)

        probabilities = checkpoint_classifier.classify_image(IMAGE_PATH, CAT_NAMES)

    elif USECASE == USECASE_TRAINING:
        # Test classification with a self trained neural network.
        # Valid architectures: 'AlexNet', 'DenseNet121', 'VGG16'
        training_classifier = FlowerClassifier.from_training_data(
            "./flowers", './checkpoints', 'VGG16', 0.001, 512, 0.2, 10,
            category_mapping=CAT_NAMES, top_k=5, gpu=True)

        training_classifier.train_network()
        accuracy = training_classifier.get_modelaccuracy()
        print(f'\nAccuracy of the network with validation data (not used before): {accuracy:.3f}\n')
        probabilities = training_classifier.classify_image(IMAGE_PATH, CAT_NAMES)
        training_classifier.save_checkpoint()

    else:
        raise ValueError("Incorrect use case")
    
    print(f'{IMAGE_PATH:<40} Probs')
    print(f'{"-" * 40} {"-" * 5}')
    for item in probabilities:
        print(f'{item["flower"]:<40} {item["probability"]:>5.3f}')
    print()
