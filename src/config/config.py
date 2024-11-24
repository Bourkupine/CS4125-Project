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
MODELS = ["random_forest"]
DEFAULT_MODEL = ["random_forest"]

#Data
DATA_PATH = 'datasets/AppGallery.csv'

#Decorators
DECORATORS = ["Duplicate", "NoiseRemover", "Translate"]
