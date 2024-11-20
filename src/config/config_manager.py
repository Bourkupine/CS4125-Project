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

    # Set a config
    def set_config(self, key, val):
        setattr(config, key, val)
        # Reload the config after editing it
        config.reload_config()

    # Return a config
    def get_config(self, key):
        return getattr(config, key, None)

    # Reload our config file after making a change
    def reload_config(self):
        importlib.reload(config)
