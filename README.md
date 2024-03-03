# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## System Setup

The solution has been developed locally with *Anaconda* based on a fork of the [Starter Code](https://github.com/udacity/aipnd-project).

Overview of the used language version, tools and libraries.

| Module           | Version |
| ---------------- | ------- |
| python           | 3.11.7  |
| conda            | 2.1.4   |
| jupyter notebook | 7.0.6   |
| matplotlib       | 3.8.0   |
| seaborn          | 0.12.2  |
| numpy            | 1.26.3  |
| pandas           | 2.1.4   |
| pillow           | 10.0.1  |
| pytorch          | 2.2.0   |
| torchvision      | 0.17.0  |

For compatibility reasons please make sure to have an environment with all components with the respective versions in place.

## Training, Test and Validation Data

Everything what you need to run the solution is in the project except the image data. This is due to the large amount of about 329MB (8189 images).

Steps to download and install the images
1. Download [Flower Data](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)
2. Create a folder ```flowers``` on the project root.
3. Unzip the files into the previously created directory

## Training and Prediction from the Command Line

import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
# from torch import __version__



if __name__ == "__main__":  
