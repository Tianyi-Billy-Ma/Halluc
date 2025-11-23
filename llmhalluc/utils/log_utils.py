import logging
from logging import DEBUG, INFO


def setup_logging(verbose):
    level = DEBUG if verbose else INFO
    logging_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=logging_format)
