from src.config import config

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

    # Return a config
    def get_config(self, key):
        return getattr(config, key, None)
