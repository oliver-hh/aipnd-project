from modules.cmdline_args import get_predict_input_args
from modules.flower_classifier import FlowerClassifier

if __name__ == "__main__":
    try:
        args = get_predict_input_args()
        #print(args)

        classifier = FlowerClassifier.from_checkpoint(args.checkpoint, args.top_k, args.gpu)

        probabilities = classifier.classify_image(args.input, args.category_names)

        # Print the prediction result of the use case
        print(f'{args.input:<40} Probs')
        print(f'{"-" * 40} {"-" * 5}')
        for item in probabilities:
            print(f'{item["flower"]:<40} {item["probability"]:>5.3f}')
        print()

    except ValueError as ve:
        print(f'Value error: {str(ve)}')
    except FileNotFoundError as fnfe:
        print(f'FileNotFoundError: {str(fnfe)}')
