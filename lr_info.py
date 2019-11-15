from keras.callbacks import Callback
import keras.backend as K

class lr_info(Callback):

    def __init__(self, model, log_mlflow=False):
        self.model = model
        self.mlflow = log_mlflow
        
    def on_epoch_end(self, epoch, logs={}):
        
        sess = K.get_session()
        
        lr = self.model.optimizer.lr
        
        cur_lr = sess.run([lr])
        print('Current learning rate: {}'.format(cur_lr[0]))

        if self.mlflow:
            import mlflow
            mlflow.log_metric('learning_rate', cur_lr[0], epoch + 1)
