from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, MaxPooling2D, Conv2D, DepthwiseConv2D, BatchNormalization, Add, ZeroPadding2D
from keras_applications import correct_pad
import keras.backend as K

def custom_backbone(input_tensor):

    # input layer
    x = ZeroPadding2D(padding=correct_pad(K, input_tensor, 3))(input_tensor)
    x = Conv2D(32, (3,3), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
        
    # blocks
    x = twoblocks(x, 128, 192, 64)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = twoblocks(x, 192, 256, 96)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = twoblocks(x, 256, 384, 128)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = twoblocks(x, 384, 512, 384)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = twoblocks(x, 512, 768, 256)
    
    return x

def twoblocks(input_layer, expand1, expand2, squeeze):
    x1 = invblock(input_layer, expand1, squeeze)
    x2 = invblock(x1, expand2, squeeze)

    return Add()([x1, x2])

def invblock(input_layer, expand, squeeze):
    """Source: https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5"""
    y = Conv2D(expand, (1,1))(input_layer)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = DepthwiseConv2D((3,3), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(squeeze, (1,1))(y)
    y = BatchNormalization()(y)
    # !!! no relu, the last is the linear bottleneck layer !!!

    return y
