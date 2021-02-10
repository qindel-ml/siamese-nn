from joblib import Parallel, delayed
from tqdm import tqdm
import os
import sys
import numpy as np

def preload_images(data, num_threads, image_dir):
    """
    Preload the images into cache.
    :param data: the list of per-image dictionaries
    :param num_threads: the number of concurrent I/O processes
    :param image_dir: the root of the images directory
    :return: an image cache dictionary
    """
    print('Caching {} images using {} threads'.format(len(data), num_threads))
    chunk_size = len(data) // num_threads
    data_chunks = [
        data[i * chunk_size:min(len(data), (i + 1) * chunk_size)
        if i < num_threads - 1 else len(data)]
        for i in range(num_threads)
    ]

    r = Parallel(n_jobs=num_threads)(
        delayed(cache_worker)
        (thread_data, image_dir, thread_cnt == num_threads - 1, num_threads)
        for thread_cnt, thread_data in enumerate(data_chunks))

    cache_list, read_fails, not_found, cache_sizes = zip(*r)

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
    print('\n\n\nTotal cache size: {:.3f} GB\n\n'.format(total_size / 1073741824))

    return img_cache


def cache_worker(data, image_dir, verbose=False, num_threads=1):
    img_cache = {}
    cnt = 0
    old_perc = 0
    old_cnt = 0
    failed = 0
    not_found = 0
    size = 0
    if verbose:
        pbar = tqdm(total=len(data * num_threads))

    for i, d in enumerate(data):
        if not os.path.exists(os.path.join(image_dir, d['path'])):
            not_found += 1
        else:
            try:
                with open(os.path.join(image_dir, d['path']), 'rb') as fr:
                    img_cache[os.path.join(image_dir, d['path'])] = fr.read()
                    size += sys.getsizeof(img_cache[os.path.join(image_dir, d['path'])])
            except:
                failed += 1
                img_cache[os.path.join(image_dir, d['path'])] = None
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

    return img_cache, failed, not_found, size
