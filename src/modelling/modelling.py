from src.models.randomforest import RandomForest
from src.modelling.data_model import Data
from src.models import model
from pandas import DataFrame

# Here we need to call the methods related to the model e.g., random forest
def model_predict(data: Data, df: DataFrame, model: model):
    model.fit(data.get_X_train(), data.get_y_train())

def model_evaluate(model, data):
    model.print_results(data)