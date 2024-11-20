from config import config
import importlib

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
        if not hasattr(self, 'init'):
            self.init = True
            self.config - self.load_config_file()

    # Load in all configs from our module as a dictionary
    def load_config_file(self):
        return {
            key: getattr(config, key) for key in dir(config)
        }

    # Set a config
    def set_config(self, key, val):
        self.config[key] = val
        config.reload_config()

    # Return a config
    def get_config(self, key):
        return self.config[key]

    # Reload our config file after making a change
    def reload_config(self):
        importlib.reload(config)
        self.config = self.load_config_file()
