from src.utils.logger_config import logger


class PPO:
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items())
        print(self.lr)