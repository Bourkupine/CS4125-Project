from src.utils.logger import log_ui_update
from src.patterns.observer import Observer


class UIUpdater(Observer):
    def update(self, email, classification):
        log_ui_update(email, classification)  # Delegate UI updates to a helper function