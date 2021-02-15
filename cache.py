from joblib import Parallel, delayed
from tqdm import tqdm
import os
import sys
import numpy as np
from PIL import Image


def preload_images(data, num_threads, image_dir, cache_flag):
    """
    Preload the images into cache.
    :param data: the list of per-image dictionaries
    :param num_threads: the number of concurrent I/O processes
    :param image_dir: the root of the images directory
    :param cache_flag: 1 if cache required 0 if not
    :return: an image cache dictionary
    """
    if cache_flag > 0:
        print('Caching and checking {} images using {} threads'.format(len(data), num_threads))
    else:
        print('Checking {} images using {} threads'.format(len(data), num_threads))
    chunk_size = len(data) // num_threads
    data_chunks = [
        data[i * chunk_size:min(len(data), (i + 1) * chunk_size)
        if i < num_threads - 1 else len(data)]
        for i in range(num_threads)
    ]

    r = Parallel(n_jobs=num_threads)(
        delayed(image_worker)
        (thread_data, image_dir, cache_flag//num_threads, thread_cnt == num_threads - 1, num_threads)
        for thread_cnt, thread_data in enumerate(data_chunks))

    image_dict, cache_list, read_fails, not_found, cache_sizes = zip(*r)

    image_dict = [x for sublist in list(image_dict) for x in sublist]

    train_imgs = {}
    for d in data:
        cur_id = d['parent_id']
        complete_path = os.path.join(image_dir, d['path'])
        if complete_path not in image_dict:
            continue
        if cur_id in train_imgs:
            train_imgs[cur_id].append(complete_path)
        else:
            train_imgs[cur_id] = [complete_path]

    total_fails = sum(read_fails)
    total_not_found = sum(not_found)
    total_size = sum(cache_sizes)
    img_cache = cache_list[0]
    for i in range(len(cache_list)):
        if i > 0:
            img_cache.update(cache_list[i])
    del cache_list
    print('Failed to read {} images ({}% of total).'.format(total_fails,
                                                            np.around(total_fails * 100 / len(data), 1)))
    print('Not found {} images ({}% of total).'.format(total_not_found,
                                                       np.around(total_not_found * 100 / len(data), 1)))
    if cache_flag > 0:
        print('\n\n\nTotal cache size: {:.3f} GB\n\n'.format(total_size / 1073741824))

    return train_imgs, img_cache


def image_worker(data, image_dir, cache_flag, verbose=False, num_threads=1):
    correct_images = []
    img_cache = {}
    cnt = 0
    old_perc = 0
    old_cnt = 0
    failed = 0
    not_found = 0
    size = 0
    cached = 0
    if verbose:
        pbar = tqdm(total=len(data * num_threads))

    for i, d in enumerate(data):
        complete_path = os.path.join(image_dir, d['path'])
        if not os.path.exists(complete_path):
            not_found += 1
            print("Not found: {}".format(complete_path))
        else:
            try:
                img = Image.open(complete_path)
                img.verify()
                img.close()
                if cache_flag > 0 and cached < cache_flag:
                    with open(complete_path, 'rb') as fr:
                        img_cache[complete_path] = fr.read()
                        file_size = sys.getsizeof(img_cache[complete_path])
                        size += file_size
                        cached += file_size
                correct_images.append(complete_path)
            except:
                failed += 1
                # img_cache[complete_path] = None
                print("Failed: {}".format(complete_path))
        cnt += 1
        if verbose:
            perc = int(cnt * 100 / len(data))
            if perc > old_perc:
                old_perc = perc
                pbar.set_description('Size {:.3f} GB, failed {}, not found {}'.format(size * num_threads / 1073741824,
                                                                                      failed * num_threads,
                                                                                      not_found * num_threads))
                pbar.update((cnt - old_cnt) * num_threads)
                old_cnt = cnt

    if verbose:
        pbar.close()

    return correct_images, img_cache, failed, not_found, size
