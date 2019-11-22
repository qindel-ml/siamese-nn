"""
This file implements an image similariy detectos. It takes a grayscale image as input and returns the probability that the images are different.
"""
def create_model(image_shape=(224, 224, 3), restart_checkpoint=None, backbone='mobilnetv2', feature_len=128, freeze=False):
    """
    Creates an image encoder.

    Args:
        image_shape: input image shape (use [None, None] for resizable network)
        restart_checkpoint: snapshot to be restored
        backbone: the backbone CNN (one of mobilenetv2, siamese, resnet50)
        feature_len: the length of the additional feature layer
        freeze: freeze the backbone
    """
    # input tensors placeholders
    from keras.layers import Input
    input_img = Input(shape=image_shape)

    # get the backbone
    backbone_name = backbone
    from keras.layers import Concatenate, GlobalMaxPool2D, GlobalAvgPool2D, BatchNormalization
    if backbone_name=='densenet121':
        print('Using DenseNet121 backbone.')
        from keras.applications.densenet import DenseNet121
        backbone = DenseNet121(input_tensor=input_img, include_top=False)
    else:
        print('Using MobileNetV2 backbone.')    
        from keras.applications.mobilenet_v2 import MobileNetV2
        backbone = MobileNetV2(input_tensor=input_img, include_top=False)

    backbone.layers.pop()
    if freeze:
        for layer in backbone.layers:
            layer.trainable = False
    backbone = backbone.output    

    from keras.layers import Dense, Activation, Flatten, MaxPooling2D
    backbone = MaxPooling2D((2, 2), 2)(backbone)
    backbone = Flatten()(backbone)
    backbone = Dense(feature_len*2)(backbone)
    backbone = BatchNormalization()(backbone)
    backbone = Activation('relu')(backbone)
    backbone = Dense(feature_len)(backbone)
    backbone = BatchNormalization()(backbone)
    backbone = Activation('sigmoid')(backbone)
    
    
    from keras.models import Model
    encoder = Model(input_img, backbone)
    
    if restart_checkpoint:
        print('Loading weights from {}'.format(restart_checkpoint))
        encoder.load_weights(restart_checkpoint, by_name=True, skip_mismatch=True)
        
    return encoder

def batch_hard_loss(outputs, loss_batch, loss_margin, soft=False, metric='euclidian'):
    """
    An implementation of the batch hard loss (eq. 5 from arXiv:1703.07737 "In Defense of the Triplet Loss for Person Re-Identification" by Hermans, Beyer & Leibe)

    Args:
       outputs: the encoded features. It is expected that the batch size of the outputs tensor is a multiple of 2 * loss_batch
       loss_batch: the size of the minibatch for the loss function
       loss_margin: the margin for the triple loss formula (m in the paper)
       soft: use soft margin, i.e. log(1+exp(distance))
       metric: 'euclidian' or 'binaryce' (binary cross entropy)
    Returns:
        the batch hard loss
    """
    import keras.backend as K
    
    # group images by examples (each example contains 2 * loss_batch images)
    examples = K.reshape(outputs, (-1, 2 * loss_batch, K.shape(outputs)[1]))

    # get the anchor image and expand the second dimension
    anchors = K.expand_dims(examples[:, 0, :], 1)

    positives = examples[:, 1:loss_batch, :] # the next loss_batch-1 images are positives
    negatives = examples[:, loss_batch: , :] # the last loss_batch images are nagatives

    if metric == 'euclidian':
        # compute the maximum positive distance
        pos_dist = K.max(K.mean(K.square(K.repeat_elements(anchors, loss_batch-1, axis=1) - positives), axis=2))

        # compute the minimum negative distance
        neg_dist = K.min(K.mean(K.square(K.repeat_elements(anchors, loss_batch, axis=1) - negatives), axis=2))
    else:
        # compute the maximum positive distance
        pos_dist = K.max(K.mean(K.binary_crossentropy(K.repeat_elements(anchors, loss_batch-1, axis=1), positives), axis=2))

        # compute the minimum negative distance
        neg_dist = K.min(K.mean(K.binary_crossentropy(K.repeat_elements(anchors, loss_batch, axis=1), negatives), axis=2))

    # compute the average true batch loss
    #import tensorflow as tf
    if not soft:
        import tensorflow as tf
        loss = K.mean(K.maximum(loss_margin + pos_dist - neg_dist, 0))
    else:
        import tensorflow as tf
        loss = K.mean(tf.math.log1p(K.exp(pos_dist - neg_dist)))

    return loss

    
