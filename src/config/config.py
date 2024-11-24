# This file contains some variable names you need to use in overall project.
#For example, this will contain the name of dataframe columns we will working on each file

# Input Columns
TICKET_SUMMARY = 'Ticket Summary'
INTERACTION_CONTENT = 'Interaction content'

# Type Columns to test
TYPE_COLS = ['Type 1', 'Type 3', 'Type 4']
CLASS_COL = 'Type 2'
GROUPED = 'Type 1'

# Models
MODELS = ["ada_boost", "linear_svc", "logistic_regression", "naive_bayes", "random_forest", "sgd"]
DEFAULT_MODEL = "random_forest"

#Data
DATA_PATH = 'datasets/AppGallery.csv'

#Decorators
DECORATORS = ["RemoveDuplicates", "NoiseRemover", "Translate"]
