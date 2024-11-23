from typing import List
from src.utils.logger import log_classification, log_metrics_update, log_ui_update

class Observer:
    def update(self, email, classification):
        # React to an email classification event
        raise NotImplementedError("Subclasses must implement the update method.")









