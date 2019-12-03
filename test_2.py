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
import shutil

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
    parser.add_argument('--output-csv', type=str, help='The CSV file where the results of the comparison will be written.')
    parser.add_argument('--paint-dir', type=str, default=None, help='The directory where the painted results will be stored. If omitted, no painting is done.')
    parser.add_argument('--exclude-auto', action='store_true', default=False, help='Do not paint an image comparison to itself.')
    parser.add_argument('--threshold', type=float, default=None, help='If set, ignores the detections below the threshold.')
    args = parser.parse_args()

    # prepare the painting dir
    if args.paint_dir:
        if not os.path.exists(args.paint_dir):
            os.makedirs(args.paint_dir)
        else:
            shutil.rmtree(args.paint_dir)
            os.makedirs(args.paint_dir)
    
    # create the model
    from model_siamese import create_model
    model, _, _ = create_model((args.image_size, args.image_size, 1), args.feature_vector_len, restart_checkpoint=args.model, backbone=args.backbone)
    
    # compare the image pairs
    if args.threshold:
        threshold = args.threshold
    else:
        threshold = 0.0
    
    results = []
    from itertools import product
    conv = 'L'

    eval_list_ = list(product(args.images1, args.images2))

    # remove symmetric evaluations
    symm = {}
    eval_list = []
    for i in range(len(eval_list_)):
        lname, rname = eval_list_[i][0], eval_list_[i][1]

        if rname in symm:
            if lname in symm[rname]:
                continue

        if not lname in symm:
            symm[lname] = {}
        symm[lname][rname] = True
        eval_list.append([lname, rname])
    
    tot_len = len(eval_list)
    pbar = tqdm(total=tot_len)
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

        if prob >= threshold:
            if args.paint_dir and ((args.exclude_auto and (x[0] != x[1])) or not(args.exclude_auto)):
                cimg_a = LetterboxImage(Image.open(x[0]))
                cimg_a.do_letterbox(600, 600, randomize_pos=False)
                cimg_b = LetterboxImage(Image.open(x[1]))
                cimg_b.do_letterbox(600, 600, randomize_pos=False)
                paint_comparison(cimg_a, os.path.basename(x[0]), cimg_b, os.path.basename(x[1]), prob, args.paint_dir)

            results.append(pd.DataFrame({'image1':x[0], 'image12':x[1], 'probability_same':prob}, index=[0]))
        pbar.update(1)
        
    results = pd.concat(results)

    results.to_csv(args.output_csv, index=False)


def paint_comparison(limg, limg_name, rimg, rimg_name, prob, output_dir):
    """
    Paints two images side by side and display the probability they are the same image.

    Args:
        limg, rimg: left and right images (numpy arrays [height, width, 3])
        limg_name, rimg_name: left and right image names
        prob: the probability they are the same image
        output_dir: the painting dir
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_width = 1

    # compute the text sizes
    txt_size_a = cv2.getTextSize(limg_name, font, font_scale, font_width)
    txt_size_b = cv2.getTextSize(rimg_name, font, font_scale, font_width)

    prob_text = 'Distance: ' + str(np.around(prob, 2))
    prob_text_size = cv2.getTextSize(rimg_name, font, font_scale, font_width)

    # compute the final image dimensions
    size = limg.width  
    sep = size // 10 # horizontal separation
    header = prob_text_size[0][1] + prob_text_size[1] + 10 # header for probability printing

    tot_width = 2 * size + sep
    tot_height = size + header

    # create the new image
    new_img = Image.new('RGB', (tot_width, tot_height), (0, 0, 0))

    # paste the images being compared
    new_img.paste(limg, (0, header))
    new_img.paste(rimg, (size + sep, header))

    pil_img = np.array(new_img)
    new_img = pil_img[:, :, ::-1].copy() 
    
    # print the image names
    cv2.putText(new_img, limg_name, (0, header + txt_size_a[0][1] + 5), font, font_scale, (255, 255, 255), font_width, cv2.LINE_AA)
    cv2.putText(new_img, rimg_name, (size + sep, header + txt_size_b[0][1] + 5), font, font_scale, (255, 255, 255), font_width, cv2.LINE_AA)

    # print the probability
    cv2.putText(new_img, prob_text, (tot_width // 2 - 200, prob_text_size[0][1] + 5), font, font_scale, (255, 255, 255), font_width, cv2.LINE_AA)
    
    # save
    file_name = os.path.join(output_dir, os.path.splitext(limg_name)[0] + '-' + os.path.splitext(rimg_name)[0] + '.jpg')
    cv2.imwrite(file_name, new_img)
    
if __name__ == "__main__":
    _main()
