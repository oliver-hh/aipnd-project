# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Project Structure and files
The project has been forked from [udacity/aipnd-project](https://github.com/udacity/aipnd-project). All folders and files that have been created or modified are listed below with a short description:


```
root_directory/  
|     
+-- modules/                  --> Self-written modules  
|   |  
|   +-- cmdline_args.py       --> Processing of command line args
|   +-- flower_classifier.py  --> Common class for training and prediction
|  
+-- protocols/  --> Output of training runs for all CNN-types
|   |               with accuracy and a test prediction
|   +-- AlexNet.txt
|   +-- DenseNet121.txt
|   +-- VGG16.txt
|  
+-- .gitignore  --> exclude folder flowers: training/testing/validation images
|               --> exclude folder checkpoints: Checkpoint files  
|  
+-- cat_to_name.json                --> Reformatted and sorted by class index,
|                                       so that there is one line per class.
|
+-- checkpoint.pth                  --> Checkoint created by Jupyter Notebook 
|
+-- Image Analysis.ipynb            --> Image characteristics, category files
|
+-- Image Classifier Project.ipynb  --> Part 1 of the project: Implement
|                                       training, accuracy determination,
|                                       load/save models and prediction.
|
+-- predict.py  --> Implementation of the prediction for flower images
+-- train.py    --> Implementation of the training of models
|
+-- README.md   --> Documentation
```

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
Both functionalities have been implemented in a single class ```FlowerClassifier``` which is intended to be instantiated in either of two ways:
1. ```from_training_data(...)```
2. ```from_checkpoint(...)```

Details for the parameters can be found in the inline documentation of ```flower_classifier.py```. This class makes it possible to keep the following scripts short and concise:
1. ```train.py```: Training neural networks
2. ```predict.py```: Classify a given image

### Training neural networks

**How to use the command line tool:**

```
usage: train.py [-h] [--save_dir SAVE_DIR] [--arch {AlexNet,DenseNet121,VGG16}] [--learning_rate LEARNING_RATE]
                [--hidden_units HIDDEN_UNITS] [--dropout_rate DROPOUT_RATE] [--epochs EPOCHS] [--gpu]
                data_dir

positional arguments:
  data_dir              Path to the folder of data

options:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Path to save trained model
  --arch {AlexNet,DenseNet121,VGG16}
                        CNN Model Architecture
  --learning_rate LEARNING_RATE
                        Learning rate for model
  --hidden_units HIDDEN_UNITS
                        Number of hidden units in model
  --dropout_rate DROPOUT_RATE
                        Dropout rate from 0 to 1
  --epochs EPOCHS       Number of epochs for training
  --gpu                 Use GPU for training if available
```

**Example of training a neural network:**

```bash
DATA_DIR='./flowers'

python train.py $DATA_DIR --arch VGG16 --epochs 5 --gpu
```

Outputs can be seen in the files of folder ```./protocols/```.

### Classify a given image

**How to use the command line tool:**

```
usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES] [--gpu] input checkpoint

positional arguments:
  input                 Path to the image to classify
  checkpoint            Path to the checkpoint file that contains the trained network

options:
  -h, --help            show this help message and exit
  --top_k TOP_K         Return top K most likely classes of the prediction
  --category_names CATEGORY_NAMES
                        File name for mapping of categories to real names
  --gpu                 Use GPU for inference if available
```

**Example of predicting an image:**

```bash
CHECKPOINT_FILE='./checkpoint.pth'
IMAGE_PATH='./flowers/test/10/image_07090.jpg'
CAT_NAMES='./cat_to_name.json'

python predict.py $IMAGE_PATH $CHECKPOINT_FILE --top_k 3
```

**Output example:**

```
./flowers/test/10/image_07090.jpg        Probs
---------------------------------------- -----
globe thistle                            0.953
artichoke                                0.040
spear thistle                            0.004
alpine sea holly                         0.001
pincushion flower                        0.001
```
