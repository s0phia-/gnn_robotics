import logging


logging.basicConfig(filename="logger.log",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG,
                    filemode='w')

logger = logging.getLogger()
