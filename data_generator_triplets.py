import numpy as np
from PIL import Image
from LetterboxImage import LetterboxImage

def data_generator(imgs, batch_size, loss_batch, input_shape, same_prob, no_aug_prob, no_augment=False, augment={}, greyscale=False):

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
        k = loss_batch
        l = batch_size // (2 * loss_batch)

        for ll in range(l):
            # store the anchor image
            if conv=='L':
                img_a = Image.open(imgs[i]).convert(conv)
            else:
                img_a = Image.open(imgs[i])

                if greyscale:
                    img_a = img_a.convert('L').convert('RGB')
                
            lbimg_a = LetterboxImage(img_a.copy())
            aug_anchor = np.random.random() >= no_aug_prob
            if aug_anchor:
                targets = lbimg_a.do_augment(augment, greyscale=greyscale)
            else:
                targets = None
            lbimg_a.do_letterbox(sizew, sizeh, randomize_pos=aug_anchor, targets=targets)
            images.append(np.array(lbimg_a) / 255.0)

            # store the augmented positive examples
            for j in range(k-1):
                lbimg_p = LetterboxImage(img_a.copy())
                aug_pos = np.random.random() >= no_aug_prob
                if aug_pos:
                    targets = lbimg_p.do_augment(augment, greyscale=greyscale)
                else:
                    targets = None
                lbimg_p.do_letterbox(sizew, sizeh, randomize_pos=aug_pos, targets=targets)
                images.append(np.array(lbimg_p) / 255.0)

            # store the negative examples

            # get a random sample of k images and make sure that the anchor is not among them
            negs_are_negs = False
            while not(negs_are_negs):
                negs = np.random.choice(imgs, k)
                negs_are_negs = not(imgs[i] in negs)

            for j in range(k):
                if conv=='L':
                    img_n = Image.open(negs[j]).convert(conv)
                else:
                    img_n = Image.open(negs[j])
                    
                    if greyscale:
                        img_n = img_n.convert('L').convert('RGB')
                        
                lbimg_n = LetterboxImage(img_n.copy())
                aug_neg = np.random.random() >= no_aug_prob
                if aug_neg:
                    targets = lbimg_n.do_augment(augment, greyscale=greyscale)
                else:
                    targets = None
                lbimg_n.do_letterbox(sizew, sizeh, randomize_pos=aug_neg, targets=targets)
                images.append(np.array(lbimg_n) / 255.0)

            # advance to next image
            i = (i + 1) % n
            if i==0:
                np.random.shuffle(imgs)
                
        # return the batch
        images = np.stack(images)
        ground_truth = np.stack(ground_truth)

        yield images, ground_truth
                               
                
