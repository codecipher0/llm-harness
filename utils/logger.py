import logging

def get_logger(name="i2i-harness"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logger