# -*- coding:utf-8 -*-
import logging
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class ConfusionMatrixCallback(Callback):
    def __init__(self, x, y, normalize_func=None):
        super(ConfusionMatrixCallback, self).__init__()
        self.x = x
        self.y = y
        self.y_pre = None
        self.normalize_func = normalize_func

    def on_epoch_end(self, epoch, logs={}):
        model = self.model
        if self.x is None or self.y is None:
            logger.warning('\n x or y is None...')
        else:
            self.y_pre = model.predict(self.x)

            def default_normalize_func(y_true, y_pre):
                _y_pre_norm = [1.0 if y[0] > 0.5 else 0.0 for y in y_pre]
                _y_true_norm = y_true.ravel()
                return _y_true_norm, _y_pre_norm
            if self.normalize_func:
                y_true_norm, y_pre_norm = self.normalize_func(self.y, self.y_pre)
            else:
                logger.warning('normalize_func is None, use default_normalize_func')
                y_true_norm, y_pre_norm = default_normalize_func(self.y, self.y_pre)
            cm = confusion_matrix(y_true_norm, y_pre_norm)
            np.set_printoptions(precision=2)
            print '\n Confusion matrix for test, without normalization'
            print cm
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print '\n Normalized confusion matrix'
            print cm_normalized
