import logging
from typing import Literal

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

log_level_map: dict[str, int]  ={
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "fatal": logging.FATAL,
    "debug": logging.DEBUG
}

def log_message(msg: str, level: str = Literal["info", "warning", "warn", "debug", "error", "fatal", "critical"]):
    if level in log_level_map:
        logger.log(level=log_level_map[level], msg=msg,)
    else: logger.log(level=logging.INFO, msg=msg,)
