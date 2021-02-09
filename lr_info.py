from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

class lr_info(Callback):

    def __init__(self, model, log_mlflow=False):
        self.model = model
        self.mlflow = log_mlflow
        
    def on_epoch_end(self, epoch, logs={}):
        
        lr = self.model.optimizer.lr
        
        cur_lr = K.eval(lr)
        print('Current learning rate: {}'.format(cur_lr))

        if self.mlflow:
            import mlflow
            mlflow.log_metric('learning_rate', cur_lr, epoch + 1)
