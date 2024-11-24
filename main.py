import random
import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from src.data.BasePreprocessor import BasePreprocessor #for remove_empty vals

from src.modelling.data_model import Data
from src.data.embeddings import get_tfidf_embd
from src.patterns.Decorators import DuplicateDecorator, NoiseRemoverDecorator, TranslateDecorator # these are adding the following functionality de_duplication, noise_remover, get_input_data, remove_empty, translate_input_data
from src.patterns.config_manager import ConfigManager
from src.utils.file_helpers import get_input_data #this is for getting the input data

seed = 0
random.seed(seed)
np.random.seed(seed)

if __name__ == '__main__':

    config_manager = ConfigManager

    models = config_manager.get_config("MODELS")
    decorator_list = config_manager.get_config("DECORATORS")

    parser = argparse.ArgumentParser()

    # Argument for passing model choice
    parser.add_argument("-m", "--model",
                    type=str,
                    required=False,
                    help="Pick which model to use for classification, use --list for available models")

    # Argument for displaying a list of available models
    parser.add_argument("--list",
                    required=False,
                    action='store_true',
                    help="List all available models")

    # Argument for entering interactive mode for choosing a model
    parser.add_argument("-i", "--interactive",
                    required=False,
                    action='store_true',
                    help="Enter interactive mode to select a model")
    
    parser.add_argument("-d", "--decorator",
                    nargs='*',
                    required=False,
                    help="Pass decorators in for preprocessing, use --dlist for available decorators")
    
    parser.add_argument("--dlist",
                    required=False,
                    action='store_true',
                    help="List all available decorators")

    args = parser.parse_args()

    if args.list:
        print(f"Current Models: {models}")
        exit()

    if args.dlist:
        print(f"Current Decorators: {decorator_list}")
        exit()

    # Code for interactively choosing model
    if args.interactive:
        print("Entered interactive mode:")
        resp = ""
        while True:
            resp = input(f"Enter model: {models}\n")
            if resp == 'quit':
                sys.exit(1)
            if resp in models:
                break
            print("Invalid choice")

    decorators = []
    if args.decorator:
        for dec in args.decorator:
            if dec not in decorator_list:
                sys.exit("Unknown decorator")
            else:
                decorators.append(dec)
        print(decorators)

    if not args.model:
        # No model passed, so we use the default model from Config
        current_model = config_manager.get_config("DEFAULT_MODEL")
    else:
        # Check if chosen model is a valid model
        if args.model in models:
            current_model = args.model
        else:
            sys.exit("Invalid model, use --list for a list of available models")


    df = get_input_data()

    interaction_content = config_manager.get_config("INTERACTION_CONTENT")
    ticket_summary = config_manager.get_config("TICKET_SUMMARY")

    df[interaction_content] = df[interaction_content].values.astype('U')
    df[ticket_summary] = df[ticket_summary].values.astype('U')

    X = get_tfidf_embd(df)
    Y = df["Type 2"].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    data = Data(X_train, X_test, Y_train, Y_test)

    "Todo: Create model and run with data"
