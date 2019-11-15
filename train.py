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
    parser.add_argument('--backbone', type=str, default='siamese', help='The network backbone: siamese(default), mobilenetv2, resnet50')
    parser.add_argument('--max-lr', type=float, default=1e-4, help='The maximum (and also initial) learning rate (1e-4 by default).')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='The minimum learning rate (1e-5 by default).')
    parser.add_argument('--lr-schedule', type=str, default='cosine', help='The learning rate schedule: cosine (default), cyclic.')
    parser.add_argument('--lr-schedule-cycle', type=int, default=100000, help='The lerning rate cycle length (number of images).')
    parser.add_argument('--images-per-epoch', type=int, default=10000, help='The number of images per epoch.')
    parser.add_argument('--start-epoch', type=int, default=1, help='The starting epoch (1 by default).')
    parser.add_argument('--end-epoch', type=int, default=5000, help='The ending epoch (5000 by default).')
    parser.add_argument('--checkpoint-name', type=str, default='chkpt', help='The root of the checkpoint names.')
    parser.add_argument('--checkpoint-freq', type=int, default=100, help='The frequency of checkpoints in epochs. Default is 100.')
    parser.add_argument('--early-stopping-patience', type=int, default=None, help='The number of epoch to wait before stopping if the validation loss does not decrease.')
    parser.add_argument('--same-prob', type=float, default=0.5, help='The probability of comparing to the same image (0.5 by default).')
    parser.add_argument('--no-aug-prob', type=float, default=0.2, help='The probability that an image is not augmented at all.')
    parser.add_argument('--crop-prob', type=float, default=0.05, help='The crop probability (0.05 by default).')
    parser.add_argument('--crop-frac', type=float, default=0.09, help='The maximum fraction of area cropped-out (0.16 by default).')
    parser.add_argument('--jitter-prob', type=float, default=0.2, help='The jitter probability (0.2 by default')
    parser.add_argument('--jitter', type=float, default=0.1, help='The jitter size (0.1 by default).')
    parser.add_argument('--rot', type=float, default=0.0, help='The rotation probability (0.0 by default).')
    parser.add_argument('--hflip', type=float, default=0.0, help='The horizontal flip probability (0.0 by default).')
    parser.add_argument('--vflip', type=float, default=0.3, help='The vertical flip probability (0.0 by default).')
    

    args = parser.parse_args()


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
    model, encoder = create_model((args.image_size, args.image_size, 1), args.feature_vector_len)

    print('\nThe model:')
    print(model.summary())

    # compile the model
    from keras.optimizers import Adam
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.max_lr))

    # prepare the callbacks
    from lr_info import lr_info
    info_lr = lr_info(model)

    if args.lr_schedule == 'cosine':
        print('Using the cosine annealing learning rate scheduler.')
        from cos_callback import CosineAnnealingScheduler
        lr_callback = CosineAnnealingScheduler(args.max_lr, args.batch_size, args.lr_schedule_cycle, min_lr=args.min_lr, verbose=True, initial_counter=(args.start_epoch - 1) * args.images_per_epoch//args.batch_size)
    else:
        from clr_callback import CyclicLR
        lr_callback = CyclicLR(model='triangular', max_lr=args.maxlr, base_lr=args.min_lr, step_size=args.lr_schedule_cycle//args.batch_size)

    from checkpoint import MyModelCheckpoint
    checkpoint = MyModelCheckpoint(
        filepath=os.path.join(args.output_dir, args.checkpoint_name + '_' + '{epoch:04d}.h5'),
        snapshot_path=os.path.join(args.output_dir, args.checkpoint_name+'.snapshot.h5'),
        model_body=encoder,
        save_best_only=do_valid,
        period=args.checkpoint_freq,
        verbose=1)

    callbacks=[info_lr, lr_callback, checkpoint]

    if do_valid and args.early_stopping_patience:
        from keras.callbacks import EarlyStopping

        callbacks.append(EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience))

    # train
    augment={
        'crop_prob':args.crop_prob,
        'crop_frac':args.crop_frac,
        'jitter_prob':args.jitter_prob,
        'jitter':args.jitter,
        'rot':args.rot,
        'hflip':args.hflip,
        'vflip':args.vflip
    }
    
    train_generator = data_generator(train_imgs,
                                     args.batch_size,
                                     (args.image_size, args.image_size, 1),
                                     args.same_prob,
                                     args.no_aug_prob,
                                     no_augment=False,
                                     augment=augment)

    if do_valid:
        val_generator = data_generator(val_imgs,
                                       args.batch_size,
                                       (args.image_size, args.image_size, 1),
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

def data_generator(imgs, batch_size, input_shape, same_prob, no_aug_prob, no_augment=False, augment={}):

    # initialize
    sizew = input_shape[0]
    sizeh = input_shape[1]
    conv = 'L' if input_shape[2] == 1 else 'RGB'

    n = len(imgs)

    # infinite loop of the generator
    i = 0
    while True:

        # prepare the next batch
        image_a = []
        image_b = []
        ground_truth = []

        b = 0
        while b < batch_size:

            # decide if the images are augmented
            do_aug =  np.random.random() >= no_aug_prob
            
            # load and letterbox the first image
            img_a = Image.open(imgs[i]).convert(conv)
            mimg_a = LetterboxImage(img_a)
            if do_aug:
                mimg_a.do_augment(augment)
            mimg_a.do_letterbox(sizew, sizeh, randomize_pos=not no_augment)
            if conv=='L':
                image_a.append(np.expand_dims(np.array(mimg_a) / 255.0, 2))
            else:
                image_a.append(np.array(mimg_a) / 255.0)

            # choose whether to load the same image or choose another
            same_img = np.random.random() < same_prob
            if same_img:
                # use the same image
                mimg_b = LetterboxImage(img_a)
            else:
                # choose the next image
                img_b = Image.open(imgs[(i + 1) % n]).convert(conv)
                mimg_b = LetterboxImage(img_b)
                i = (i + 1) %n

            # letterbox the second image
            if do_aug:
                mimg_b.do_augment(augment)
            mimg_b.do_letterbox(sizew, sizeh, randomize_pos=not no_augment)
            if conv=='L':
                image_b.append(np.expand_dims(np.array(mimg_b) / 255.0, 2))
            else:
                image_b.append(np.array(mimg_b) / 255.0)

            # add the similarity indicator to the ground truths
            ground_truth.append(int(same_img))

            # increment the counters with wraparound
            i = (i + 1) %n
            b += 1

            # if wrapping around, reshuffle images
            if i==0:
                np.random.shuffle(imgs)

        # return the batch
        image_a = np.stack(image_a)
        image_b = np.stack(image_b)
        ground_truth = np.stack(ground_truth)

        yield [image_a, image_b], ground_truth
                               
                

#############################

if __name__ == "__main__":
    _main()
