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
import pandas as pd
from data_generator import data_generator

##############################

def _main():

    # argument parsing
    parser = argparse.ArgumentParser(description='Test the similarity of lists of image.')
    parser.add_argument('--model', type=str, help='The trained model file.')
    parser.add_argument('--images1', nargs='+', type=str, help='The first image list.')
    parser.add_argument('--images2', nargs='+', type=str, help='The second image list.')
    parser.add_argument('--image-size', type=int, default=224, help='The image size in pixels, default is 224 (meaning 224x224).')
    parser.add_argument('--feature-vector-len', type=int, default=1024, help='The length of the feature vector (1024 by default).')
    parser.add_argument('--backbone', type=str, default='siamese', help='The network backbone: siamese(default), mobilenetv2, resnet50')
    parser.add_argument('--output-csv', type=str, help='A CSV file where the results of the comparison will be written.')

    args = parser.parse_args()


    # create the image lists
    print(args.images1)
    print(args.images2)

    

    # create the model
    from model import create_model
    model, model_body, encoder = create_model((args.image_size, args.image_size, 1), args.feature_vector_len, restart_checkpoint=args.model, backbone=args.backbone)

        

if __name__ == "__main__":
    _main()
