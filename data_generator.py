import numpy as np
from PIL import Image
from AugmentedLetterbox import AugmentedLetterbox

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

            # decide if both images are left unaugmented
            do_aug_both = np.random.random() >= no_aug_prob
            
            # decide if the left image is augmented
            do_aug_l =  (np.random.random() >= no_aug_prob) and do_aug_both
            
            # load and letterbox the first image
            img_a = Image.open(imgs[i]).convert(conv)
            mimg_a = AugmentedLetterbox(img_a)
            mimg_a.letterbox(sizew, sizeh, randomize_pos=not no_augment, augments = augment if do_aug_l else None)
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
                mimg_b = AugmentedLetterbox(img_b)
                i = (i + 1) %n

            # letterbox the second image
            # decide if the right image is augmented
            do_aug_r =  (np.random.random() >= no_aug_prob) and do_aug_both

            mimg_b.letterbox(sizew, sizeh, randomize_pos=not no_augment, augments = augment if do_aur_r else None)
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
                               
                
