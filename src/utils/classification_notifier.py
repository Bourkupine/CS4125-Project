from typing import List
from observer import Observer

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