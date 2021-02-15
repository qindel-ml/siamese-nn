#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from data_generator_triplets import data_generator
from model_triplet import create_model, batch_hard_loss
from lr_info import lr_info
from cos_callback import CosineAnnealingScheduler
from clr_callback import CyclicLR
from checkpoint import MyModelCheckpoint
from load_data import load_data
from tensorflow import config
from cache import preload_images

physical_devices = config.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], True)


def _main():
    # argument parsing
    parser = argparse.ArgumentParser(description='Trains an image similarity detector.')
    parser.add_argument('--training-images-dir', type=str, help='The directory containing the training images'
                                                                'input files (JSON).')
    parser.add_argument('--validation-images-dir', type=str, default=None,
                        help='The directory containing the validation images input files.'
                             'If not specified, than no validation is performed (default behavior).')
    parser.add_argument('--images-dir', type=str, help='The root of the images directory.')
    parser.add_argument('--output-dir', type=str, help='The output directory where the checkpoints will be stored.')
    parser.add_argument('--restart-checkpoint', type=str, default=None, help='The checkpoint from which to restart.')
    parser.add_argument('--image-size', type=int, default=224,
                        help='The image size in pixels, default is 224 (meaning 224x224).')
    parser.add_argument('--preload-images', type=int, default=0,
                        help='Preload (cache) images before starting training, 0 if not needed, else: number of bytes '
                             'to load in cache.')
    parser.add_argument('--greyscale', type=int, default=0, help='If set to 1, converts images to greyscale.')
    parser.add_argument('--batch-size', type=int, default=24, help='The training minibatch size.')
    parser.add_argument('--loss-batch', type=int, default=4, help='The loss minibatch size.')
    parser.add_argument('--backbone', type=str, default='mobilenetv2',
                        help='The network backbone: mobilenetv2 (default), densenet121')
    parser.add_argument('--freeze-backbone', type=int, default=0, help='If set to 1, freeze the backbone.')
    parser.add_argument('--feature-len', type=int, default=128,
                        help='If larger than 0, a 1x1 convolution is added that converts the backbone output features '
                             'to a layer with depth equal to --feature-len.')
    parser.add_argument('--margin', type=float, default=0.4, help='The margin for the triple loss (default is 0.4).')
    parser.add_argument('--soft', type=int, default=0, help='If set to 1, use soft margins when computing loss.')
    parser.add_argument('--metric', type=str, default='euclidian',
                        help='The distance metric: Euclidian (euclidian) or binary cross-entropy (binaryce). By '
                             'fedault it is Euclidian.')
    parser.add_argument('--max-lr', type=float, default=1e-4,
                        help='The maximum (and also initial) learning rate (1e-4 by default).')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='The minimum learning rate (1e-5 by default).')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        help='The learning rate schedule: cosine (default), cyclic.')
    parser.add_argument('--lr-schedule-cycle', type=int, default=100000,
                        help='The lerning rate cycle length (number of images).')
    parser.add_argument('--images-per-epoch', type=int, default=10000, help='The number of images per epoch.')
    parser.add_argument('--start-epoch', type=int, default=1, help='The starting epoch (1 by default).')
    parser.add_argument('--end-epoch', type=int, default=5000, help='The ending epoch (5000 by default).')
    parser.add_argument('--checkpoint-name', type=str, default='chkpt', help='The root of the checkpoint names.')
    parser.add_argument('--checkpoint-freq', type=int, default=100,
                        help='The frequency of checkpoints in epochs. Default is 100.')
    parser.add_argument('--early-stopping-patience', type=int, default=-1,
                        help='The number of epoch to wait before stopping if the validation loss does not decrease. '
                             'Set to -1 to disable (default)')
    parser.add_argument('--no-aug-prob', type=float, default=0.2,
                        help='The probability that an image is not augmented at all.')
    parser.add_argument('--crop-prob', type=float, default=0.0, help='The crop probability (0.05 by default).')
    parser.add_argument('--crop-frac', type=float, default=0.09,
                        help='The maximum fraction of area cropped-out (0.16 by default).')
    parser.add_argument('--fill-letterbox', type=int, default=0, help='Fill the letterbox (for small images')
    parser.add_argument('--jitter-prob', type=float, default=0.2, help='The jitter probability (0.2 by default')
    parser.add_argument('--jitter', type=float, default=0.1, help='The jitter size (0.1 by default).')
    parser.add_argument('--rotation-prob', type=float, default=0.0, help='The rotation probability.')
    parser.add_argument('--rotation-angle', type=float, default=0.0, help='The maximum rotation angle.')
    parser.add_argument('--rotation-expand-prob', type=float, default=0,
                        help='Probability to expand the image when rotating to not lose anything.')
    parser.add_argument('--scale-prob', type=float, default=0.1, help='The rescaling probability.')
    parser.add_argument('--scale-min', type=float, default=1.0, help='The minimum image rescaling factor.')
    parser.add_argument('--scale-max', type=float, default=1.0, help='The maximum image rescaling factor.')
    parser.add_argument('--hflip', type=float, default=0.0, help='The horizontal flip probability (0.0 by default).')
    parser.add_argument('--no-colour-transforms', type=int, default=0, help='Do not transform colors.')
    parser.add_argument('--vflip', type=float, default=0.0, help='The vertical flip probability (0.0 by default).')
    parser.add_argument('--hue', type=float, default=0.05, help='The hue variation (ignored for siamese backbone).')
    parser.add_argument('--sat', type=float, default=0.2,
                        help='The saturation variation (ignored for siamese backbone).')
    parser.add_argument('--val', type=float, default=0.2, help='The value variation (ignored for siamese backbone).')
    parser.add_argument('--mlflow', type=int, default=0,
                        help='Set to 1 if using MLflow. Metrics and artifacts will be logged.')

    args = parser.parse_args()

    # start the mlflow autologging
    if args.mlflow:
        import mlflow.keras
        mlflow.keras.autolog()

    # create the training image list
    train_data = load_data(args.training_images_dir, verbose=False)
    train_imgs, train_cache = preload_images(train_data, 4, args.images_dir, args.preload_images)
    train_parents = list(train_imgs.keys())
    np.random.shuffle(train_parents)

    train_lens = {}
    for k, v in train_imgs.items():
        cur_len = len(v)
        if cur_len in train_lens:
            train_lens[cur_len] += 1
        else:
            train_lens[cur_len] = 1
    train_lens = pd.DataFrame(train_lens, index=[0])
    print("Training length distribution:")
    print(train_lens)

    if args.validation_images_dir:
        do_valid = True
        val_data = load_data(args.validation_images_dir, verbose=False)
        val_imgs, val_cache = preload_images(val_data, 4, args.images_dir, args.preload_images)
        val_parents = list(val_imgs.keys())
        np.random.shuffle(val_parents)

    else:
        do_valid = False

    print('There are {} training images.'.format(len(train_imgs)))
    if do_valid:
        print('There are {} validation images.'.format(len(val_imgs)))

    # create the output directory if necessary
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # scale the larning rate to the batch size
    max_lr = args.max_lr
    min_lr = args.min_lr

    # create the model
    num_channels = 1 if args.backbone == 'siamese' else 3
    encoder = create_model((args.image_size, args.image_size, num_channels), restart_checkpoint=args.restart_checkpoint,
                           backbone=args.backbone, feature_len=args.feature_len, freeze=args.freeze_backbone == 1)

    # compile the model with the initial learning rate
    bh_loss = Lambda(batch_hard_loss, output_shape=(1,), name='batch_hard',
                     arguments={'loss_batch': args.loss_batch, 'loss_margin': args.margin, 'soft': args.soft == 1,
                                'metric': args.metric})(encoder.output)
    model = Model(encoder.input, bh_loss)
    model.compile(loss={'batch_hard': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=max_lr))
    print(model.summary())

    print('Loss metric: {}'.format(args.metric))
    if args.soft == 1:
        print('Using soft margins.')

    # prepare the callbacks
    info_lr = lr_info(model, args.mlflow == 1)

    # learning rate
    true_batch_size = args.batch_size // args.loss_batch

    print('Scaling the learning rate minimum to {} and maximum (initial) to {}'.format(min_lr, max_lr))
    if args.lr_schedule == 'cosine':
        print('Using the cosine annealing learning rate scheduler.')
        lr_callback = CosineAnnealingScheduler(max_lr, true_batch_size, args.lr_schedule_cycle, min_lr=min_lr,
                                               verbose=True,
                                               initial_counter=(args.start_epoch - 1) * args.images_per_epoch)
    else:
        lr_callback = CyclicLR(mode='triangular', max_lr=max_lr, base_lr=min_lr,
                               step_size=args.lr_schedule_cycle // true_batch_size)

    # checkpoints
    checkpoint = MyModelCheckpoint(
        filepath=os.path.join(args.output_dir, args.checkpoint_name + '_' + '{epoch:04d}'),
        snapshot_path=os.path.join(args.output_dir, args.checkpoint_name + '-snapshot'),
        model_body=None,
        encoder=encoder,
        save_best_only=do_valid,
        period=args.checkpoint_freq,
        verbose=1,
        mlflow=args.mlflow == 1)

    callbacks = [info_lr, lr_callback, checkpoint]

    if do_valid and args.early_stopping_patience != -1:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience))

    # train
    print('Batch configuration:')
    print('Loss batch: {}'.format(args.loss_batch))
    print('Positives + anchors: {}'.format(args.loss_batch // 4))
    print('Negatives: {}'.format(args.loss_batch - args.loss_batch // 4))
    print('Effective minibatch: {}'.format(true_batch_size))
    print('Encoder minibatch: {}'.format(args.batch_size))

    augment = {
        'scale_prob': args.scale_prob,
        'scale_min': args.scale_min,
        'scale_max': args.scale_max,
        'crop_prob': args.crop_prob,
        'crop_frac': args.crop_frac,
        'jitter_prob': args.jitter_prob,
        'jitter': args.jitter,
        'rotate_prob': args.rotation_prob,
        'rotate_angle': args.rotation_angle,
        'rotate_expand_prob': args.rotation_expand_prob,
        'hflip_prob': args.hflip,
        'vflip_prob': args.vflip
    }
    if args.no_colour_transforms == 0:
        augment['hue']: args.hue
        augment['saturation']: args.sat
        augment['value']: args.val

    train_generator = data_generator(train_imgs,
                                     train_parents,
                                     args.batch_size,
                                     args.loss_batch,
                                     (args.image_size, args.image_size, num_channels),
                                     args.no_aug_prob,
                                     augment=augment,
                                     greyscale=args.greyscale == 1,
                                     fill_letterbox=args.fill_letterbox == 1,
                                     cache=train_cache)

    if do_valid:
        val_generator = data_generator(val_imgs,
                                       val_parents,
                                       args.batch_size,
                                       args.loss_batch,
                                       (args.image_size, args.image_size, num_channels),
                                       args.no_aug_prob,
                                       augment=augment,
                                       greyscale=args.greyscale == 1,
                                       fill_letterbox=args.fill_letterbox == 1,
                                       cache=val_cache)
    else:
        val_generator = None

    model.fit_generator(train_generator,
                        steps_per_epoch=max(1, args.images_per_epoch // true_batch_size),
                        validation_data=val_generator,
                        validation_steps=max(1, args.images_per_epoch // true_batch_size),
                        epochs=args.end_epoch,
                        initial_epoch=args.start_epoch - 1,
                        callbacks=callbacks)


##############################


#############################

if __name__ == "__main__":
    _main()
