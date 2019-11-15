"""
This class is a modified keras.callbacks.ModelCheckpoint. It accepts several additional parameters and saves the checkpoint regardless of the monitor.
"""


import os
from keras.callbacks import Callback
import tensorflow as tf
import shutil
import json
import numpy as np

class MyModelCheckpoint(Callback):

    def __init__(self,
                 model_body,
                 filepath,
                 snapshot_path,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 mode='auto',
                 period=2,
                 absolute_counter=True,
                 mlflow=False):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.model_body = model_body
        self.filepath = filepath
        self.snapshot_path = snapshot_path
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.absolute_counter = absolute_counter
        self.mlflow = mlflow
        
        if mode not in ['auto', 'min', 'max']:
            logging.warning('MyModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'
            
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        saved = False

        if (not self.absolute_counter and self.epochs_since_last_save >= self.period) or (self.absolute_counter and (epoch+1)%self.period==0):
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
        else:
            filepath = self.snapshot_path

        if True:
            if self.save_best_only:
                current = logs.get(self.monitor)

                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))

                        self.best = current
                        saved = True
                        self.model_body.save(filepath, overwrite=True)

                    else:
                            
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
                            
            else:
                    
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    saved = True
                    self.model_body.save(filepath, overwrite=True)


        if self.mlflow and self.save_best_only:
            import mlflow
            mlflow.log_metric("best_val_loss", self.best, epoch)
                            
        if saved:

            if self.mlflow:
                import mlflow
                mlflow.log_artifact(filepath)
                if not(self.eval_dump is None):
                    mlflow.log_artifact(jsonname)
                
