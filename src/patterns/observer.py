from typing import List
from src.utils.logger import log_classification, log_metrics_update, log_ui_update


class ClassificationNotifier:

    #A hub to manage observers and notify them about classification events.
    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer):
        #Attach an observer to the notifier.
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer):
        #Detach an observer from the notifier.
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, email, classification):
        #Notify all attached observers about a classification event.
        for observer in self._observers:
            observer.update(email, classification)


class Observer:
    def update(self, email, classification):
        # React to an email classification event
        raise NotImplementedError("Subclasses must implement the update method.")


class Logger(Observer):
    def update(self, email, classification):
        # Delegate logging to a helper function
        log_classification(email, classification)


class MetricsTracker(Observer):
    def __init__(self):
        self.stats = {"Spam": 0, "Not Spam": 0}

    def update(self, email, classification):
        self.stats[classification] += 1
        log_metrics_update(self.stats)  # Delegate metrics logging to a helper function


class UIUpdater(Observer):
    def update(self, email, classification):
        log_ui_update(email, classification)  # Delegate UI updates to a helper function
