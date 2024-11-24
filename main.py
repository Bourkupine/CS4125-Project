import argparse
import random
import sys
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.modelling.data_model import Data
from src.models.factories.AdaBoostFactory import AdaBoostFactory
from src.models.factories.LinearSVCFactory import LinearSVCFactory
from src.models.factories.LogisticRegressionFactory import LogisticRegressionFactory
from src.models.factories.ModelFactory import ModelFactory
from src.models.factories.NaiveBayesFactory import NaiveBayesFactory
from src.models.factories.RandomForestFactory import RandomForestFactory
from src.models.factories.SGDFactory import SGDFactory
from src.patterns.config_manager import ConfigManager
from src.preprocessing.BasePreprocessor import BasePreprocessor
from src.preprocessing.Decorators.DuplicateDecorator import DuplicateDecorator
from src.preprocessing.Decorators.NoiseRemoverDecorator import NoiseRemoverDecorator
from src.preprocessing.Decorators.TranslateDecorator import TranslateDecorator
from src.preprocessing.embeddings import get_tfidf_embd
from src.utils.classification_notifier import ClassificationNotifier
from src.utils.file_helpers import get_input_data
from src.utils.logger import Logger
from src.utils.metrics import MetricsTracker

seed = 0
random.seed(seed)
np.random.seed(seed)

if __name__ == '__main__':

    config_manager = ConfigManager()

    #Initialize notifier and attach observers
    notifier = ClassificationNotifier()
    notifier.attach(MetricsTracker())
    notifier.attach(Logger())

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set the log level (e.g., DEBUG, INFO, WARNING)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()  # Send logs to the terminal
        ]
    )

    models = config_manager.get_config("MODELS")
    decorator_list = config_manager.get_config("DECORATORS")

    model_map = {
        "ada_boost": AdaBoostFactory(),
        "linear_svc": LinearSVCFactory(),
        "logistic_regression": LogisticRegressionFactory(),
        "naive_bayes": NaiveBayesFactory(),
        "random_forest": RandomForestFactory(),
        "sgd": SGDFactory(),
    }

    decorator_map = {
        "RemoveDuplicates": DuplicateDecorator,
        "NoiseRemover": NoiseRemoverDecorator,
        "Translate": TranslateDecorator,
    }

    preprocessor = BasePreprocessor()
    model = None

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
    decorators = sorted(decorators, key={"RemoveDuplicates": 0, "Translate": 1, "NoiseRemover": 2}.get)
    for dec in decorators:
        preprocessor = decorator_map[dec](preprocessor)

    if not args.model:
        # No model passed, so we use the default model from Config
        current_model = config_manager.get_config("DEFAULT_MODEL")
    else:
        # Check if chosen model is a valid model
        if args.model in models:
            current_model = args.model
        else:
            sys.exit("Invalid model, use --list for a list of available models")

    model_factory: ModelFactory = model_map[current_model]
    model = model_factory.create_model(model_name=current_model, load_saved=True)

    df = get_input_data()

    df = preprocessor.preprocess(df)

    X = get_tfidf_embd(df)
    Y = df["Type 2"].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    data = Data(X_train, X_test, Y_train, Y_test)


    model.train(data, save=True)

    results = model.predict(X_test)
    
    for idx, result in enumerate(results):
        emailid = idx
        classification = result
        notifier.notify(emailid, classification)
