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
from tqdm import tqdm

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

    # create the model
    from model import create_model
    model, _, _ = create_model((args.image_size, args.image_size, 1), args.feature_vector_len, restart_checkpoint=args.model, backbone=args.backbone)
    
    # compare the image pairs
    results = []
    from itertools import product
    conv = 'L'

    eval_list = list(product(args.images1, args.images2))
    
    pbar = tqdm(total=len(eval_list))
    pbar.set_description('Comparing...')
    for x in eval_list:
        img_a = Image.open(x[0]).convert(conv)
        mimg_a = LetterboxImage(img_a)
        mimg_a.do_letterbox(args.image_size, args.image_size, randomize_pos=False)

        img_b = Image.open(x[1]).convert(conv)
        mimg_b = LetterboxImage(img_b)
        mimg_b.do_letterbox(args.image_size, args.image_size, randomize_pos=False)

        input_a = np.expand_dims(np.expand_dims(np.array(mimg_a) / 255.0, 2), 0)
        input_b = np.expand_dims(np.expand_dims(np.array(mimg_b) / 255.0, 2), 0)

        prob = model.predict([input_a, input_b])[0, 0]

        results.append(pd.DataFrame({'image1':x[0], 'image12':x[1], 'probability_same':prob}, index=[0]))
        pbar.update(1)
        
    results = pd.concat(results)

    results.to_csv(args.output_csv, index=False)
    
if __name__ == "__main__":
    _main()
