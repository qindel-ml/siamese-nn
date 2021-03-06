#!/usr/bin/env python3

# silence tensorflow and Keras warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# imports
from keras import backend as K
import cv2
from PIL import Image
import argparse
import numpy as np
from LetterboxImage import LetterboxImage
from data_generator import data_generator

##############################

def _main():

    # argument parsing
    parser = argparse.ArgumentParser(description='Trains an image similarity detector.')
    parser.add_argument('--training-images-dir', type=str, help='The training images directory.')
    parser.add_argument('--validation-images-dir', type=str, default=None, help='The validation images directory. If not specified, than no validation is performed (defualt behavior).')
    parser.add_argument('--output-dir', type=str, help='The output directory where the checkpoints will be stored.')
    parser.add_argument('--restart-checkpoint', type=str, default=None, help='The checkpoint from which to restart.')
    parser.add_argument('--image-size', type=int, default=224, help='The image size in pixels, default is 224 (meaning 224x224).')
    parser.add_argument('--batch-size', type=int, default=8, help='The training minibatch size.')
    parser.add_argument('--feature-vector-len', type=int, default=1024, help='The length of the feature vector (1024 by default).')
    parser.add_argument('--use-l2', type=int, default=0, help='If set to 1, use L2 instead of L1 difference.')
    parser.add_argument('--backbone', type=str, default='siamese', help='The network backbone: siamese(default), mobilenetv2, resnet50')
    parser.add_argument('--freeze-backbone', type=int, default=0, help='Set to 1 to freeze the backbone (0 by default).')
    parser.add_argument('--max-lr', type=float, default=1e-4, help='The maximum (and also initial) learning rate (1e-4 by default).')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='The minimum learning rate (1e-5 by default).')
    parser.add_argument('--lr-schedule', type=str, default='cosine', help='The learning rate schedule: cosine (default), cyclic.')
    parser.add_argument('--lr-schedule-cycle', type=int, default=100000, help='The lerning rate cycle length (number of images).')
    parser.add_argument('--images-per-epoch', type=int, default=10000, help='The number of images per epoch.')
    parser.add_argument('--start-epoch', type=int, default=1, help='The starting epoch (1 by default).')
    parser.add_argument('--end-epoch', type=int, default=5000, help='The ending epoch (5000 by default).')
    parser.add_argument('--checkpoint-name', type=str, default='chkpt', help='The root of the checkpoint names.')
    parser.add_argument('--checkpoint-freq', type=int, default=100, help='The frequency of checkpoints in epochs. Default is 100.')
    parser.add_argument('--early-stopping-patience', type=int, default=-1, help='The number of epoch to wait before stopping if the validation loss does not decrease. Set to -1 to disable (default)')
    parser.add_argument('--same-prob', type=float, default=0.5, help='The probability of comparing to the same image (0.5 by default).')
    parser.add_argument('--no-aug-prob', type=float, default=0.2, help='The probability that an image is not augmented at all.')
    parser.add_argument('--crop-prob', type=float, default=0.05, help='The crop probability (0.05 by default).')
    parser.add_argument('--crop-frac', type=float, default=0.09, help='The maximum fraction of area cropped-out (0.16 by default).')
    parser.add_argument('--jitter-prob', type=float, default=0.2, help='The jitter probability (0.2 by default')
    parser.add_argument('--jitter', type=float, default=0.1, help='The jitter size (0.1 by default).')
    parser.add_argument('--rot', type=float, default=0.0, help='The rotation probability (0.0 by default).')
    parser.add_argument('--hflip', type=float, default=0.0, help='The horizontal flip probability (0.0 by default).')
    parser.add_argument('--vflip', type=float, default=0.3, help='The vertical flip probability (0.0 by default).')
    parser.add_argument('--hue', type=float, default=0.05, help='The hue variation (ignored for siamese backbone).')
    parser.add_argument('--sat', type=float, default=0.2, help='The saturation variation (ignored for siamese backbone).')
    parser.add_argument('--val', type=float, default=0.2, help='The value variation (ignored for siamese backbone).')
    parser.add_argument('--mlflow', type=int, default=0, help='Set to 1 if using MLflow. Metrics and artifacts will be logged.')

    args = parser.parse_args()

    # start the mlflow autologging
    if args.mlflow:
        import mlflow.keras
        mlflow.keras.autolog()

    
    # create the image lists
    exts = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.gif', '.GIF', '.tiff', '.TIFF', '.TIF', '.bmp', '.BMP')
    train_imgs = []
    train_dir_files = os.listdir(args.training_images_dir)
    
    for f in train_dir_files:
        if f.endswith(exts):
            train_imgs.append(os.path.join(args.training_images_dir, f))

        np.random.shuffle(train_imgs)
    
    if args.validation_images_dir:
        do_valid = True
        val_imgs = []
        val_dir_files = os.listdir(args.validation_images_dir)
        
        for f in val_dir_files:
            if f.endswith(exts):
                val_imgs.append(os.path.join(args.validation_images_dir, f))

        np.random.shuffle(val_imgs)
    else:
        do_valid = False

    print('There are {} training images.'.format(len(train_imgs)))
    if do_valid:
        print('There are {} validation images.'.format(len(val_imgs)))

    # create the output directory if necessary
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)  

    # create the model
    from model import create_model
    num_channels = 1 if args.backbone == 'siamese' else 3
    model, model_body, encoder = create_model((args.image_size, args.image_size, num_channels), args.feature_vector_len, restart_checkpoint=args.restart_checkpoint, backbone=args.backbone, freeze=args.freeze_backbone==1, l2=args.use_l2==1)

    print('\nThe model:')
    print(model.summary())

    # prepare the callbacks
    from lr_info import lr_info
    info_lr = lr_info(model, args.mlflow==1)

    # learning rate

    # scale the larning rate to the batch size
    max_lr = args.max_lr * np.sqrt(args.batch_size)
    min_lr = args.min_lr * np.sqrt(args.batch_size)

    print('Scaling the learning rate minimum to {} and maximum (initial) to {}'.format(min_lr, max_lr))
    print('The original values are {} and {}, and are multiplied by the root of the batch size {}.'.format(args.min_lr, args.max_lr, args.batch_size))

    if args.lr_schedule == 'cosine':
        print('Using the cosine annealing learning rate scheduler.')
        from cos_callback import CosineAnnealingScheduler
        lr_callback = CosineAnnealingScheduler(max_lr, args.batch_size, args.lr_schedule_cycle, min_lr=min_lr, verbose=True, initial_counter=(args.start_epoch - 1) * args.images_per_epoch//args.batch_size)
    else:
        from clr_callback import CyclicLR
        lr_callback = CyclicLR(model='triangular', max_lr=maxlr, base_lr=min_lr, step_size=args.lr_schedule_cycle//args.batch_size)

    # checkpoints
    from checkpoint import MyModelCheckpoint
    checkpoint = MyModelCheckpoint(
        filepath=os.path.join(args.output_dir, args.checkpoint_name + '_' + '{epoch:04d}'),
        snapshot_path=os.path.join(args.output_dir, args.checkpoint_name+'-snapshot'),
        model_body=model_body,
        encoder = encoder,
        save_best_only=do_valid,
        period=args.checkpoint_freq,
        verbose=1,
        mlflow= args.mlflow==1)

    callbacks=[info_lr, lr_callback, checkpoint]

    if do_valid and args.early_stopping_patience != -1:
        from keras.callbacks import EarlyStopping

        callbacks.append(EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience))

    # compile the model with the initial learning rate
    from keras.optimizers import Adam
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=max_lr))
        
    # train
    augment={
        'crop_prob':args.crop_prob,
        'crop_frac':args.crop_frac,
        'jitter_prob':args.jitter_prob,
        'jitter':args.jitter,
        'rot':args.rot,
        'hflip':args.hflip,
        'vflip':args.vflip,
        'hue':args.hue,
        'sat':args.sat,
        'val':args.val
    }
    
    train_generator = data_generator(train_imgs,
                                     args.batch_size,
                                     (args.image_size, args.image_size, num_channels),
                                     args.same_prob,
                                     args.no_aug_prob,
                                     no_augment=False,
                                     augment=augment)

    if do_valid:
        val_generator = data_generator(val_imgs,
                                       args.batch_size,
                                       (args.image_size, args.image_size, num_channels),
                                       args.same_prob,
                                       args.no_aug_prob,
                                       no_augment=False,
                                       augment=augment)
    else:
        val_generator = None

    model.fit_generator(train_generator,
                        steps_per_epoch=max(1, args.images_per_epoch//args.batch_size),
                        validation_data=val_generator,
                        validation_steps=max(1, args.images_per_epoch//args.batch_size),
                        epochs=args.end_epoch,
                        initial_epoch=args.start_epoch-1,
                        callbacks=callbacks)
        
##############################
                

#############################

if __name__ == "__main__":
    _main()
