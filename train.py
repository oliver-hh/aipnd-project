from modules.cmdline_args import get_train_input_args
from modules.flower_classifier import FlowerClassifier

if __name__ == "__main__":  
    try:
        args = get_train_input_args()
        #print(args)

        classifier = FlowerClassifier.from_training_data(
            args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units,
            args.dropout_rate, args.epochs, args.gpu)
        
        classifier.train_model()
        
    except ValueError as ve:
        print(f'Value error: {str(ve)}')
    except FileNotFoundError as fnfe:
        print(f'FileNotFoundError: {str(fnfe)}')
    