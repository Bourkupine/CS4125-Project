from config import config

'''
Singleton Pattern:
'''

class ConfigManager:
    instance = None

    def __new__(cls):
        # If an instance doesn't exist, create a new one
        if cls.instance is None:
            cls.instance = super(ConfigManager, cls).__new__(cls)
        return cls.instance
    
    def __init__(self):
        if not hasattr(self, 'config'):
            self.config = {}

    # Set a config
    def set_config(self, key, val):
        self.config[key] = val

    # Return a config
    def get_config(self, key):
        return self.config[key]
