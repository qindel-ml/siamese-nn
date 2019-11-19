import numpy as np
from PIL import Image
from LetterboxImage import LetterboxImage

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
        images = []
        ground_truth = [1] * batch_size
        k = (batch_size - 1) // 2

        # store the anchor image
        img_a = Image.open(imgs[i]).convert(conv)
        lbimg_a = LetterboxImage(img_a.copy())
        lbimg_a.do_letterbox(sizew, sizeh, randomize_pos=False)
        images.append(np.array(lbimg_a) / 255.0)

        # store the augmented positive examples
        for j in range(k):
            lbimg_p = LetterboxImage(img_a.copy())
            lbimg_p.do_augment(augment)
            lbimg_p.do_letterbox(sizew, sizeh, randomize_pos=True)
            images.append(np.array(lbimg_p) / 255.0)

        # store the negative examples
        wraparound = False
        for j in range(k):
            jj = (i + j + 1) % n
            img_n = Image.open(imgs[jj]).convert(conv)
            lbimg_n = LetterboxImage(img_n)
            lbimg_n.do_augment(augment)
            lbimg_n.do_letterbox(sizew, sizeh, randomize_pos=True)
            images.append(np.array(lbimg_n) / 255.0)


        i = (i + 1) % n
        if i==0:
            np.random.shuffle(imgs)
            wraparound = False
                
        # return the batch
        images = np.stack(images)
        ground_truth = np.stack(ground_truth)

        yield images, ground_truth
                               
                
