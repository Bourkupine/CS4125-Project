from src.patterns.observer import Observer
from src.utils.logger import log_metrics_update


class MetricsTracker(Observer):
    def __init__(self):
        self.stats = {"Spam": 0, "Not Spam": 0}

    def update(self, email, classification):
        self.stats[classification] += 1
        log_metrics_update(self.stats)  # Delegate metrics logging to a helper function