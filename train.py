from modules.cmdline_args import get_train_input_args

if __name__ == "__main__":  
    try:
        args = get_train_input_args()
        print(args)
    except ValueError as ve:
        print(f'Value error: {str(ve)}')
    except FileNotFoundError as fnfe:
        print(f'FileNotFoundError: {str(fnfe)}')
    