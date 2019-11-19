"""
This file implements an image similariy detectos. It takes a grayscale image as input and returns the probability that the images are different.
"""
def create_model(image_shape=(224, 224, 3), restart_checkpoint=None, backbone='mobilnetv2'):
    """
    Creates an image encoder.

    Args:
        image_shape: input image shape (use [None, None] for resizable network)
        restart_checkpoint: snapshot to be restored
        backbone: the backbone CNN (one of mobilenetv2, siamese, resnet50)
    """
    # input tensors placeholders
    from keras.layers import Input
    input_img = Input(shape=image_shape)

    # get the backbone
    backbone_name = backbone
    from keras.layers import Concatenate, GlobalMaxPool2D, GlobalAvgPool2D
    if backbone_name=='resnet50':
        raise Exception('ResNet50 backbone not implemented!')
        print('Using ResNet50 backbone.')
        from keras.applications.resnet import ResNet50
        backbone = ResNet50(input_tensor=image_img, include_top=False)
    else:
        print('Using MobileNetV2 backbone.')    
        from keras.applications.mobilenet_v2 import MobileNetV2
        backbone = MobileNetV2(input_tensor=input_img, include_top=False)

    backbone = Concatenate(axis=-1)([GlobalMaxPool2D()(backbone.output), GlobalAvgPool2D()(backbone.output)])
        
    # join networks
    from keras.models import Model
    encoder = Model(input_img, backbone)
    
    if restart_checkpoint:
        print('Loading weights from {}'.format(restart_checkpoint))
        encoder.load_weights(restart_checkpoint, by_name=True, skip_mismatch=True)
        
    return encoder

def batch_hard_loss(outputs, loss_batch, loss_margin):
    """
    An implementation of the batch hard loss (eq. 5 from arXiv:1703.07737 "In Defense of the Triplet Loss for Person Re-Identification" by Hermans, Beyer & Leibe)

    Args:
       outputs: the encoded features. It is expected that the batch size of the outputs tensor is 2 * loss_batch + 1
       loss_batch: the size of the minibatch for the loss function
       loss_margin: the margin for the triple loss formula (m in the paper)

    Returns:
        the batch hard loss
    """
    import keras.backend as K
    
    # get the anchor, positive and negative features
    #anchors = K.repeat_elements(K.expand_dims(outputs[0, :], 0), rep=loss_batch, axis=0) # the first image is the anchor
    feat_len = K.cast(K.shape(outputs)[1], K.dtype(outputs))
    anchors = outputs[0, :] / feat_len
    positives = outputs[1:1+loss_batch, :] / feat_len # the next loss_batch_size images are positives
    negatives = outputs[1+loss_batch:, :] / feat_len # the last loss_batch_size images are nagatives

    # compute the maximum positive distance
    pos_dist = K.max(K.sqrt(K.sum(K.square(anchors - positives), axis=1)))

    # compute the minimum negative distance
    neg_dist = K.min(K.sqrt(K.sum(K.square(anchors - negatives), axis=1)))

    # compute loss
    loss = K.maximum(loss_margin + pos_dist - neg_dist, 0)
    return loss

    
