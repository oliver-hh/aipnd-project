"""_summary_
Module flower_classifier classifies flowers by a freshly trained network or 
a loaded network. It containts the following classes and methods:
- Class FlowerClassifier
"""
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
                 training_path=None, arch=None, learning_rate=None, hidden_units=None, epochs=None):

        # Initializae common parameters
        self.category_mapping = category_mapping
        self.top_k = top_k
        self.gpu = gpu

        # Initialize specific parameters
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
            # Load the model from the checkpoint
        elif (training_path is not None and arch is not None and learning_rate is not None and
              hidden_units is not None and epochs is not None):
            self.training_path = training_path
            self.arch = arch
            self.learning_rate = learning_rate
            self.hidden_units = hidden_units
            self.epochs = epochs
            # Train the model
        else:
            raise ValueError("Incorrect parameters")

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
            _type_: _description_
        """
        return cls(training_path=training_path, arch=arch, learning_rate=learning_rate,
                   hidden_units=hidden_units, epochs=epochs,
                   category_mapping=category_mapping, top_k=top_k, gpu=gpu)

if __name__ == "__main__":  

    checkpoint_classifier = FlowerClassifier.from_checkpoint(
        "path/to/checkpoint", category_mapping="path/to/mapping", top_k=5, gpu=True)

    trained_classifier = FlowerClassifier.from_training_data(
        "path/to/training/data", "vgg16", 0.01, 100, 10, 
        category_mapping="path/to/mapping", top_k=5, gpu=True)
