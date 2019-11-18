""" 
Based on https://github.com/4uiiurz1/keras-cosine-annealing/blob/master/cosine_annealing.py
"""

import math
from keras.callbacks import Callback
from keras import backend as K


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, initial_lr, batch_size, half_period, min_lr=None, verbose=False, initial_counter=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.initial_lr = initial_lr
        if min_lr is None:
            self.min_lr = self.initial_lr / 100.0
        else:
            self.min_lr = min_lr
        self.batch_size = batch_size
        self.half_period = half_period / self.batch_size
        self.counter = initial_counter
        self.x = min([math.pi, (self.counter / self.half_period) * math.pi])
        self.lr = self.min_lr + (self.initial_lr - self.min_lr) * (1.0 + math.cos(self.x)) / 2.0

        if verbose:
            print('Cosine annealing scheduler half-period: {} batches.'.format(self.half_period))
            print('Cosine annealing scheduler min LR: {}, max LR: {}.'.format(self.min_lr, self.initial_lr))
            print('Current learning rate: {}'.format(self.lr))
            
    def on_batch_end(self, batch, logs=None):

        self.counter += 1
        
        if self.counter%50 == 0:
        
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')

            self.x = (self.counter / self.half_period) * math.pi
            self.lr = self.min_lr + (self.initial_lr - self.min_lr) * (1.0 + math.cos(self.x)) / 2.0
            K.set_value(self.model.optimizer.lr, self.lr)


