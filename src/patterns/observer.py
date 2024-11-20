from typing import List


class EmailClassifier:
    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer):
        # Attach an observer
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer):
        #Detach an observer.
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, email, classification):
        #Notify all observers of a classification event.
        for observer in self._observers:
            observer.update(email, classification)

    def classify(self, email):
        #Simulate email classification and notify observers.
        # Placeholder for classification logic
        classification = "Spam" if "offer" in email.lower() else "Not Spam"
        print(f"Classified '{email}' as {classification}")

        # Notify observers
        self.notify(email, classification)

class Observer:
    def update(self, email, classification):
        #React to an email classification event.
        raise NotImplementedError("Subclasses must implement the update method.")

class Logger(Observer):
    def update(self, email, classification):
        print(f"[Logger] Email: '{email}' classified as {classification}")

class MetricsTracker(Observer):
    def __init__(self):
        self.stats = {"Spam": 0, "Not Spam": 0}

    def update(self, email, classification):
        self.stats[classification] += 1
        print(f"[MetricsTracker] Updated stats: {self.stats}")

class UIUpdater(Observer):
    def update(self, email, classification):
        print(f"[UIUpdater] Dashboard updated: '{email}' -> {classification}")

