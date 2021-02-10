import numpy as np
from PIL import Image
import io
from AugmentedLetterbox import AugmentedLetterbox


def data_generator(imgs, parents, batch_size, loss_batch, input_shape, same_prob, no_aug_prob,
                   no_augment=False, augment={}, greyscale=False, fill_letterbox=False,
                   cache = None):

    # initialize
    sizew = input_shape[0]
    sizeh = input_shape[1]
    conv = 'L' if input_shape[2] == 1 else 'RGB'
    has_cache = cache is not None

    n = len(parents)
    np.random.shuffle(parents)
    # infinite loop of the generator
    i = 0
    while True:

        # prepare the next batch
        images = []
        ground_truth = [1] * batch_size
        k = loss_batch
        l = batch_size // loss_batch

        for ll in range(l):
            # store the positive examples (the anchor is randomly chosen among them)
            cur_pos = np.random.choice(imgs[parents[i]], k//4)

            for j in range(k//4):
                cur_path = cur_pos[j]
                if has_cache:
                    if conv == 'L':
                        img_a = Image.open(io.BytesIO(cache[cur_path])).convert('RGB').convert(conv)
                    else:
                        img_a = Image.open(io.BytesIO(cache[cur_path])).convert('RGB')
                else:
                    if conv == 'L':
                        img_a = Image.open(cur_path).convert(conv)
                    else:
                        img_a = Image.open(cur_path)

                if greyscale:
                    img_a = img_a.convert('L').convert('RGB')
                lbimg_a = AugmentedLetterbox(img_a.copy())
                aug_pos = np.random.random() >= no_aug_prob
                lbimg_a.letterbox(sizew, sizeh, randomize_pos=aug_pos,
                              augments=augment if aug_pos else None, fill_letterbox=fill_letterbox)
                images.append(np.array(lbimg_a) / 255.0)

            # get a random sample of k images and make sure that the anchor is not among them
            negs_are_negs = False
            while not negs_are_negs:
                negs = np.random.choice(parents, k)
                negs_are_negs = not(parents[i] in negs)

            # store the negative examples
            for j in range(k - k//4):
                cur_path = np.random.choice(imgs[negs[j]], 1)[0]
                if has_cache:
                    if conv == 'L':
                        img_n = Image.open(io.BytesIO(cache[cur_path])).convert('RGB').convert(conv)
                    else:
                        img_n = Image.open(io.BytesIO(cache[cur_path])).convert('RGB')
                else:
                    if conv == 'L':
                        img_n = Image.open(cur_path).convert(conv)
                    else:
                        img_n = Image.open(cur_path)

                if greyscale:
                    img_n = img_n.convert('L').convert('RGB')
                lbimg_n = AugmentedLetterbox(img_n.copy())
                aug_neg = np.random.random() >= no_aug_prob
                lbimg_n.letterbox(sizew, sizeh, randomize_pos=aug_neg,
                                  augments=augment if aug_neg else None, fill_letterbox=fill_letterbox)
                images.append(np.array(lbimg_n) / 255.0)

            # advance to next image
            i = (i + 1) % n
            if i == 0:
                np.random.shuffle(parents)
                
        # return the batch
        images = np.stack(images)
        ground_truth = np.stack(ground_truth)

        yield images, ground_truth
