# Copyright (c) OpenMMLab. All rights reserved.
import glob
import logging
import os
import sys
import traceback
from typing import Callable, Optional

import multiprocess as mp


def target_wrapper(target: Callable,
                   log_level: int,
                   ret_value: Optional[mp.Value] = None,
                   *args,
                   **kwargs):
    """The wrapper used to start a new subprocess.

    Args:
        target (Callable): The target function to be wrapped.
        log_level (int): Log level for logging.
        ret_value (mp.Value): The success flag of target.

    Return:
        Any: The return of target.
    """
    logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')
    logger.level
    logger.setLevel(log_level)
    if ret_value is not None:
        ret_value.value = -1
    try:
        result = target(*args, **kwargs)
        if ret_value is not None:
            ret_value.value = 0
        return result
    except Exception as e:
        logging.error(e)
        traceback.print_exc(file=sys.stdout)


mmdeploy_logger = None


def get_root_logger(log_file=None, log_level=logging.INFO) -> logging.Logger:
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        logging.Logger: The obtained logger
    """
    try:
        from mmcv.utils import get_logger
        logger = get_logger(
            name='mmdeploy', log_file=log_file, log_level=log_level)
    except Exception:
        global mmdeploy_logger
        if mmdeploy_logger is not None:
            return mmdeploy_logger
        import logging
        logger = logging.getLogger('mmdeploy')

        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]
        file_mode = 'w'

        if log_file is not None:
            file_handler = logging.FileHandler(log_file, file_mode)
            handlers.append(file_handler)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)

        logger.setLevel(log_level)

        mmdeploy_logger = logger
    return logger


def get_file_path(prefix, candidates) -> str:
    """Search for file in candidates.

    Args:
        prefix (str): Prefix of the paths.
        cancidates (str): Candidate paths
    Returns:
        str: file path or '' if not found
    """
    for candidate in candidates:
        wildcard = os.path.abspath(os.path.join(prefix, candidate))
        paths = glob.glob(wildcard)
        if paths:
            lib_path = paths[0]
            return lib_path
    return ''
