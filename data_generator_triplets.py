import numpy as np
from PIL import Image
import io
from AugmentedLetterbox import AugmentedLetterbox


def process_image(cache, conv, cur_path, greyscaled, no_aug_prob,
                  sizew, sizeh, augment, fill_letterbox):
    # Checking if file is needed to be read from cache or from disk
    if cur_path in cache.keys():
        if conv == 'L':
            img = Image.open(io.BytesIO(cache[cur_path])).convert('RGB').convert(conv)
        else:
            img = Image.open(io.BytesIO(cache[cur_path])).convert('RGB')
    else:
        if conv == 'L':
            img = Image.open(cur_path).convert(conv)
        else:
            img = Image.open(cur_path)
    if greyscaled:
        img = img.convert('L').convert('RGB')
    lbimg = AugmentedLetterbox(img.copy())
    aug = np.random.random() >= no_aug_prob
    lbimg.letterbox(sizew, sizeh, randomize_pos=aug,
                    augments=augment if aug else None, fill_letterbox=fill_letterbox)
    return lbimg


def data_generator(imgs, parents, batch_size, loss_batch, input_shape, no_aug_prob, augment={},
                   greyscale=False, fill_letterbox=False, cache={}):
    # initialize
    sizew = input_shape[0]
    sizeh = input_shape[1]
    conv = 'L' if input_shape[2] == 1 else 'RGB'

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
            cur_pos = np.random.choice(imgs[parents[i]], k // 4)

            for j in range(k // 4):
                cur_path = cur_pos[j]
                img_a = \
                    process_image(cache, conv, cur_path, greyscale, no_aug_prob,
                                  sizew, sizeh, augment, fill_letterbox)
                images.append(np.array(img_a) / 255.0)

            # get a random sample of k images and make sure that the anchor is not among them
            negs_are_negs = False
            while not negs_are_negs:
                negs = np.random.choice(parents, k)
                negs_are_negs = not (parents[i] in negs)

            # store the negative examples
            for j in range(k - k // 4):
                cur_path = np.random.choice(imgs[negs[j]], 1)[0]
                img_n = \
                    process_image(cache, conv, cur_path, greyscale, no_aug_prob,
                                  sizew, sizeh, augment, fill_letterbox)
                images.append(np.array(img_n) / 255.0)

            # advance to next image
            i = (i + 1) % n
            if i == 0:
                np.random.shuffle(parents)

        # return the batch
        images = np.stack(images)
        ground_truth = np.stack(ground_truth)

        yield images, ground_truth
