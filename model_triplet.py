from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, GlobalMaxPool2D, \
    GlobalAvgPool2D, Concatenate, Multiply, Average
from tensorflow.keras.models import Model
from custom_backbone import custom_backbone


def create_model(
        image_shape=(224, 224, 3),
        restart_checkpoint=None,
        backbone='mobilnetv2',
        feature_len=128,
        freeze=False
):
    """
    Creates an image encoder.

    Args:
        image_shape: input image shape (use [None, None] for resizable network)
        restart_checkpoint: snapshot to be restored
        backbone: the backbone CNN (one of mobilenetv2, densent121, custom)
        feature_len: the length of the additional feature layer
        freeze: freeze the backbone
    """
    input_img = Input(shape=image_shape)

    # add the backbone
    backbone_name = backbone

    if backbone_name == 'densenet121':
        print('Using DenseNet121 backbone.')
        backbone = DenseNet121(
            input_tensor=input_img,
            include_top=False
        )
        backbone.layers.pop()
        if freeze:
            for layer in backbone.layers:
                layer.trainable = False
        backbone = backbone.output
    elif backbone_name == 'mobilenetv2':
        print('Using MobileNetV2 backbone.')
        backbone = MobileNetV2(
            input_tensor=input_img,
            include_top=False
        )
        backbone.layers.pop()
        if freeze:
            for layer in backbone.layers:
                layer.trainable = False
        backbone = backbone.output
    elif backbone_name == 'custom':
        backbone = custom_backbone(input_tensor=input_img)
    else:
        raise Exception('Unknown backbone: {}'.format(backbone_name))

        # add the head layers
    gmax = GlobalMaxPool2D()(backbone)
    gavg = GlobalAvgPool2D()(backbone)
    gmul = Multiply()([gmax, gavg])
    ggavg = Average()([gmax, gavg])
    backbone = Concatenate()([gmax, gavg, gmul, ggavg])
    backbone = BatchNormalization()(backbone)
    backbone = Dense(feature_len)(backbone)
    backbone = Activation('sigmoid')(backbone)

    encoder = Model(input_img, backbone)

    if restart_checkpoint:
        print('Loading weights from {}'.format(restart_checkpoint))
        encoder.load_weights(restart_checkpoint, by_name=True, skip_mismatch=True)

    return encoder


def batch_hard_loss(outputs, loss_batch, loss_margin, soft=False, metric='euclidian'):
    """
    An implementation of the batch hard loss (eq. 5 from arXiv:1703.07737 "In Defense of the Triplet Loss for Person
    Re-Identification" by Hermans, Beyer & Leibe)

    Args:
       outputs: the encoded features. We expecte that the batch size of the outputs tensor is a multiple of loss_batch
       loss_batch: the size of the minibatch for the loss function
       loss_margin: the margin for the triple loss formula (m in the paper)
       soft: use soft margin, i.e. log(1+exp(distance))
       metric: 'euclidian' or 'binaryce' (binary cross entropy)
    Returns:
        the batch hard loss
    """
    import tensorflow.keras.backend as K

    # group images by examples (each example contains loss_batch images)
    examples = K.reshape(outputs, (-1, loss_batch, K.shape(outputs)[1]))

    # get the anchor image and expand the second dimension
    anchors = K.expand_dims(examples[:, 0, :], 1)

    positives = examples[:, 1:loss_batch // 4, :]  # the next loss_batch-1 images are positives
    negatives = examples[:, loss_batch // 4:, :]  # the last loss_batch images are nagatives

    if metric == 'euclidian':
        # compute the maximum positive distance
        pos_dist = \
            K.max(
                K.mean(
                    K.square(
                        K.repeat_elements(anchors, loss_batch // 4 - 1, axis=1) - positives
                    ),
                    axis=2
                )
            )

        # compute the minimum negative distance
        neg_dist = \
            K.min(
                K.mean(
                    K.square(
                        K.repeat_elements(anchors, loss_batch - loss_batch // 4, axis=1) - negatives
                    ),
                    axis=2
                )
            )
    else:
        # compute the maximum positive distance
        pos_dist = \
            K.max(
                K.mean(
                    K.binary_crossentropy(
                        K.repeat_elements(anchors, loss_batch // 4 - 1, axis=1), positives
                    ),
                    axis=2
                )
            )

        # compute the minimum negative distance
        neg_dist = \
            K.min(
                K.mean(
                    K.binary_crossentropy(
                        K.repeat_elements(anchors, loss_batch - loss_batch // 4, axis=1), negatives
                    ), axis=2
                )
            )

    # compute the average true batch loss
    if not soft:
        loss = K.mean(K.maximum(loss_margin + pos_dist - neg_dist, 0))
    else:
        import tensorflow as tf
        loss = K.mean(tf.math.log1p(K.exp(pos_dist - neg_dist)))

    return loss


