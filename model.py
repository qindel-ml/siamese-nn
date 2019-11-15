"""
This file implements an image similariy detectos. It takes a grayscale image as input and returns the probability that the images are different.
"""
def create_model(image_shape=(224, 224, 1), feature_vector_len=1024, restart_checkpoint=None, backbone='siamese'):
    """
    Creates a siamese model. 

    Args:
        image_shape: input image shape (use [None, None] for resizable network)
        restart_checkpoint: snapshot to be restored
        backbone: the backbone CNN (one of mobilenetv2, siamese, resnet50)
    """
    # input tensors placeholders
    from keras.layers import Input
    input_a = Input(shape=(image_shape[0], image_shape[1], 1))
    input_b = Input(shape=(image_shape[0], image_shape[1], 1))

    # get the backbone
    if backbone=='siamese':
        print('Using siamese backbone.')
        from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten
        from keras.models import Sequential

        backbone = Sequential()
        backbone.add(Conv2D(64, (10,10), activation='relu', input_shape=image_shape, name='conv2D_1'))
        backbone.add(BatchNormalization(name='BN_1'))
        backbone.add(MaxPooling2D(name='MaxPool_1'))

        backbone.add(Conv2D(128, (7,7), activation='relu', name='conv2D_2'))
        backbone.add(BatchNormalization(name='BN_2'))
        backbone.add(MaxPooling2D(name='MaxPool_2'))

        backbone.add(Conv2D(128, (4,4), activation='relu', name='conv2D_3'))
        backbone.add(BatchNormalization(name='BN_3'))
        backbone.add(MaxPooling2D(name='MaxPool_3'))

        backbone.add(Conv2D(256, (4,4), activation='relu', name='conv2D_4'))
        backbone.add(BatchNormalization(name='BN_4'))
        backbone.add(MaxPooling2D(name='MaxPool_4'))

        backbone.add(Flatten(name='Flatten'))
        backbone.add(Dense(feature_vector_len, activation='sigmoid', name='Features'))
        backbone.add(BatchNormalization(name='BN_5'))

    elif backbone=='resnet50':
        raise Exception('ResNet50 backbone not implemented!')
        print('Using ResNet50 backbone.')
        from keras.applications.resnet import ResNet50
    else:
        raise Exception('MobileNetV2 backbone not implemented!')
        print('Using MobileNetV2 backbone.')    
        from keras.applications.mobilenet_v2 import MobileNetV2

        
    # join networks
    print('The encoder backbone:')
    print(backbone.summary())
    from keras.models import Model
    encoder = Model(input_a, backbone(input_a))
    # load the backbone weights

    encoded_a = backbone(input_a)
    encoded_b = backbone(input_b)

    # similarity prediction
    from keras.layers import Lambda, Dense
    import keras.backend as K

    # distance
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_a, encoded_b])
    
    # prediction
    prediction = Dense(1,activation='sigmoid', name='sigmoid_final')(L1_distance)

    # final model
    model = Model(inputs=[input_a, input_b],outputs=prediction)
    model_body = Model(inputs=[input_a, input_b],outputs=prediction)
    
    if restart_checkpoint:
        print('Loading weights from {}'.format(restart_checkpoint + '-weights.h5'))
        model.load_weights(restart_checkpoint + '-weights.h5', by_name=True, skip_mismatch=True)
        
    return model, model_body, encoder
