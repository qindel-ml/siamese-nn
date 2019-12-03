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

import keras
import tqdm
import numpy as np
from PIL import Image
import argparse
import cloudpickle
import os
from tqdm import tqdm
from Letterbox import Letterbox
from encode_utils import encode_images, shortest

def _main():
    parser = argparse.ArgumentParser(description='Compares two images: computes their distance.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='image1', type=str, help='First image.')
    parser.add_argument(dest='image2', type=str, help='Second image.')
    parser.add_argument(dest='model_path', type=str, help='The path to Keras model,')
    parser.add_argument('--image-size', type=int, default=224, help='The image size.')
    args = parser.parse_args()

    enc = encode_images(args.model_path, [args.image1, args.image2])
    dist, _, _, _ = shortest(enc[0]['embeddings'], enc[1]['embeddings'])
    dist2, _, _, _ = shortest(enc[1]['embeddings'], enc[0]['embeddings'])
    print('Distance: {} ({})'.format(dist, dist2))
    
if __name__ == "__main__":
    _main()
