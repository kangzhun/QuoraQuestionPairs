# -*- coding:utf-8 -*-
import os
import logging
from logging.handlers import TimedRotatingFileHandler

from config import LOG_PATH, LOG_FILE_NAME, LOG_BACKUP_COUNT


class BaseLogger(object):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        fh = TimedRotatingFileHandler(filename=os.path.join(LOG_PATH, LOG_FILE_NAME), when='D',
                                      backupCount=LOG_BACKUP_COUNT)
        fh.setLevel(logging.DEBUG)

        fm = logging.Formatter('%(asctime)s - [%(session_id)s] - %(levelname)s - %(message)s')
        fh.setFormatter(fm)

        if not self.logger.handlers:
            self.logger.addHandler(fh)

        self.logger.propagate = False

        self.session_id = kwargs.get("session_id", "default-session")

    def log_base(self, level, msg, *args, **kwargs):
        log_handler_map = {
            "debug": self.logger.debug,
            "info": self.logger.info,
            "warn": self.logger.warn,
            "error": self.logger.error,
            "exception": self.logger.exception
        }
        kwargs["extra"] = {"session_id": self.session_id}
        msg_str = msg
        if not isinstance(msg_str, str):
            if isinstance(msg_str, unicode):
                msg_str = str(msg_str.encode('utf-8'))
            else:
                msg_str = str(msg_str)
        if args:
            msg_str = msg % args
        msg_list = msg_str.split(os.linesep)
        for line in msg_list:
            log_handler_map[level](line, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.log_base("debug", msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log_base("info", msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.log_base("warn", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log_base("error", msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.log_base("exception", msg, *args, **kwargs)
