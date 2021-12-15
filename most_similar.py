#!/usr/bin/env python3

import numpy as np
import cloudpickle
import cv2
from PIL import Image
from Letterbox import Letterbox
from tqdm import tqdm
import shutil
import argparse
import pandas as pd
import os
from encode_utils import shortest

def _main():
    parser = argparse.ArgumentParser(
        description='Creates a table containing the most similar images in an embeddings database for each image in the test embeddings set.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest='embedding',
        help='The path to a .pickle file containing the embeddings database.'
    )
    parser.add_argument(
        dest='test',
        help='The past to a .pickle file containing the embeddings to be tested.'
    )
    parser.add_argument(dest='output_csv', help='The output CSV file.')
    parser.add_argument('--top-k', type=int, default=5, help='The top-k closesimages will be found.')
    parser.add_argument('--batch-size', type=int, default=300, help='The batch size for comparison.')
    parser.add_argument('--paint-dir', type=str, default=None, help='Painted comparison directory.')
    args = parser.parse_args()

    # load embeddings and create the arrays
    with open(args.embedding, 'rb') as inpf:
        embs_ = cloudpickle.load(inpf)
    with open(args.test, 'rb') as inpf:
        tembs_ = cloudpickle.load(inpf)

    # detect if the two files are the same file
    auto_compare = args.embedding==args.test
    if auto_compare:
        print('Auto-comparing!')
    
    embs = []
    images = []
    num_embs = len(embs_[0]['embeddings'])
    for e in embs_:
        for i in range(len(e['embeddings'])):
            embs.append(e['embeddings'][i])
            images.append(e['image'])
    embs = np.stack(embs)
    images = np.array(images)

    tembs = []
    timages = []
    num_tembs = len(tembs_[0]['embeddings'])
    for e in tembs_:
        for i in range(len(e['embeddings'])):
            tembs.append(e['embeddings'][i])
            timages.append(e['image'])
    tembs = np.stack(tembs)
    timages = np.array(timages)
    
    distances = []
    col_names = ['test_image'] \
        + ['closest_image_'+str(i) for i in range(args.top_k)] \
        + ['closest_dist_'+str(i) for i in range(args.top_k)] 
    pbar = tqdm(total=len(tembs_))
    # loop over all images
    for i in range(len(tembs_)):
        closest_images = []
        
        pivot = tembs[i*num_tembs : (i+1)*num_tembs, :]
        
        # loop over all batches
        for k in range(embs.shape[0]//args.batch_size + 1):
            if k * args.batch_size < embs.shape[0]:

                # get the current batch
                slice_min = k * args.batch_size
                slice_max = min([(k + 1) * args.batch_size, embs.shape[0]])

                slice_list = list(range(slice_min, slice_max))

                # skip auto comparison
                if auto_compare:
                    rem = list(range(i*num_tembs, (i+1)*num_tembs))
                    slice_list = list(set(slice_list) - set(rem))
                
                batch = embs[slice_list, :]
                batch_images = images[slice_list]

                # get the top-k * num_embeddings(database) * num_embeddings(test)
                # smallest distances and their indices
                _, _, cur_dists, cur_idxs = shortest(pivot, batch, args.top_k * num_embs * num_tembs)

                # get the image list corresponding to the shortest distances
                cur_images = [batch_images[ii[1]] for ii in cur_idxs]

                # create a new data frame containing the images and their distances
                tmp_array = np.concatenate([cur_images, cur_dists])
                tmp_array = tmp_array.reshape(2, args.top_k * num_embs * num_tembs).transpose()
                tmp_df = pd.DataFrame(tmp_array, columns=['image', 'distance'])
                tmp_df['distance'] = tmp_df['distance'].astype(float)


                # sort all by distance, group by image and select the closest row for each image
                tmp_df.sort_values(by='distance', inplace=True)
                tmp_df = tmp_df.groupby(['image']).head(1).head(args.top_k)

                closest_images.append(tmp_df)

        # join all batch data frames, sort by distance and pick top_k
        closest_df = pd.concat(closest_images)
        closest_df.sort_values(by='distance', inplace=True)
        closest_df = closest_df.iloc[:args.top_k, :]

        # create the data frame to add to the final result
        imgs_array = closest_df['image'].values
        dists_array = closest_df['distance'].values
        test_array = np.array([timages[i*num_tembs]])
        tmp_array = np.concatenate([test_array, imgs_array, dists_array])
        tmp_df = pd.DataFrame(np.expand_dims(tmp_array, axis=0), columns=col_names)
        distances.append(tmp_df)
        pbar.update(1)

    pbar.close()

    distances = pd.concat(distances)
    distances.to_csv(args.output_csv, index=False)
            
if __name__ == "__main__":
    _main()
