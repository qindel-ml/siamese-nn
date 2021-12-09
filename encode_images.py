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

import tqdm
import numpy as np
from PIL import Image
import argparse
import cloudpickle
import os
from tqdm import tqdm
from Letterbox import Letterbox
from encode_utils import encode_images

def _main():
    parser = argparse.ArgumentParser(description='Encodes images in a directory.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='images_dir', type=str, help='The images directory.')
    parser.add_argument(dest='model_path', type=str, help='The path to Keras model,')
    parser.add_argument(dest='output_file', type=str, help='The output .pickle file.')
    parser.add_argument('--image-size', type=int, default=224, help='The image size.')
    parser.add_argument('--only-hor', action='store_true', default=False, help='Only do horizontal flips.')
    parser.add_argument('--fill', action='store_true', default=False, help='Zoom to fill letterbox if the image is small.')
    args = parser.parse_args()

    images = os.listdir(args.images_dir)
    full_paths = [os.path.join(args.images_dir, img) for img in images]
    embeddings = encode_images(
        model_path=args.model_path,
        images=full_paths,
        letterbox_size=args.image_size,
        verbose=True,
        onlyhor=args.only_hor,
        fill=args.fill
    )

    with open(args.output_file, 'wb') as of:
        cloudpickle.dump(embeddings, of)

if __name__ == "__main__":
    _main()
