import json
from pathlib import Path


class ConfigProvider:
    @staticmethod
    def get_config(config_file):
        with open(config_file) as file:
            config = json.load(file)
            return config
