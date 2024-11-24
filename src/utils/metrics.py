from src.patterns.observer import Observer
from src.utils.logger import log_metrics_update

class MetricsTracker(Observer):
    def __init__(self):
        # Initialize an empty dictionary to dynamically track classifications.
        self.stats = {}

    def update(self, email, classification):
        # Increment the count for the given classification, initializing if necessary.
        if classification not in self.stats:
            self.stats[classification] = 0  # Initialize if classification is new.
        self.stats[classification] += 1  # Increment count for this classification.

        log_metrics_update(self.stats)  # Delegate metrics logging to a helper function