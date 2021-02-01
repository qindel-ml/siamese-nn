from PIL import Image
from Letterbox import Letterbox
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

def encode_images(model_path, images, letterbox_size=224, verbose=False, onlyhor=False, fill=False):
    """
    Encodes images and returns a list of embeddings.
    
    Args:
        model: a path to an encoder model that accepts an RGB image and returns a 1D vector.
        images: a list of image paths
        letterbox_size: the letterbox dimension (in pixels)
        verbose: show encoding progress
        onlyhor: only use horizontal flips
        fill: zoom to fill letterbox
        
    Returns:
        a list of 2D arrays (8, num_features), 
        each array row corresponding to an image orientation
    """
    
    model = load_model(model_path)
    return encode_(
        model=model,
        images=images,
        letterbox_size=letterbox_size,
        verbose=verbose,
        onlyhor=onlyhor,
        fill=fill
    )

def encode_(model, images, letterbox_size, verbose, onlyhor=False, fill=False):
    """
    Encodes images and returns a list of embeddings.
    
    Args:
        model: a Keras encoder model that accepts an RGB image and returns a 1D vector.
        images: a list of image paths
        letterbox_size: the letterbox dimension (in pixels)
        verbose: show encoding progress
        onlyhor: only use horizontal flips
        fill: zoom to fill letterbox
        
    Returns:
        a list of 2D arrays (8, num_features), 
        each array row corresponding to an image orientation
    """
    if verbose:
        pbar = tqdm(total=len(images))   
    results = []
    for image in images:
        orig_img = Image.open(image)
        lbimgs = []
        for hflip in [False, True]:
            for vflip in list(set([False, True and not(onlyhor)])):
                for rot in list(set([False, True and not(onlyhor)])):
                    cur_img = orig_img.copy()                    
                    if hflip:
                        cur_img = cur_img.transpose(Image.FLIP_LEFT_RIGHT)
                    if vflip:
                        cur_img = cur_img.transpose(Image.FLIP_TOP_BOTTOM)
                    if rot:
                        cur_img = cur_img.transpose(Image.ROTATE_90)
                    lbimg = Letterbox(cur_img)
                    lbimg.letterbox(
                        sizeh=letterbox_size,
                        sizew=letterbox_size,
                        randomize_pos=False,
                        fill_letterbox=fill
                    )
                    lbimgs.append(np.array(lbimg) / 255.0)
        lbimgs = np.stack(lbimgs)
        img_results = model.predict(lbimgs)
        if verbose:
            pbar.update(1)
        results.append({'image':image, 'embeddings':img_results})
    if verbose:
        pbar.close()
    return results

def shortest(emb1, emb2, top_k=5):
    """
    Computest the shortest distance between the two arrays of embeddings.
    
    Args:
        emb1, emb2: 2D numpy arrays (num_el, num_features) to be compares
        top_k: the number of closes images to keep
    Returns:
        the shortest distance
    """
    
    # use numpy transformations to avoid loops
    x1 = np.expand_dims(emb1, axis=1) # insexrt dummy scond dimension (num_el1, num_features) -> (num_el1, 1, num_features)
    
    # replicate the second embedings array num_el1 times:
    # [e0, e1, ..., en] -> [e0, e1, ..., en, e0, e1, ..., en, <n-3 more times>, e0, e1, ..., en]

    # (num_el2, num_features) -> (num_el2, 1, num_features)
    x2 = np.expand_dims(emb2, axis=1)

    # (num_el2, 1, num_features) -> (num_el2, num_el1, num_features)
    x2 = np.repeat(x2, emb1.shape[0], axis=1)

    # (num_el2, num_el1, num_features) -> (num_features, num_el1, num_el2)
    x2 = x2.transpose()

     # (num_features, num_el1, num_el2) -> (num_features, num_el1*num_el2)
    x2 = x2.reshape(x1.shape[2], emb1.shape[0]*emb2.shape[0])

    # (num_features, num_el1*num_el2) -> (num_el1 * num_el2, num_features)
    x2 = x2.transpose()
    
    # sqdiff[i, j] contains the differences between ith row of emb1 and jth row of emb2
    sqdiff = np.mean(np.square(x1 - x2), axis=2)[0:emb1.shape[0], 0:emb2.shape[0]]

    # get the minimum and top-k indices and distances
    min_idx = np.unravel_index(np.argmin(sqdiff), (emb1.shape[0], emb2.shape[0]))
    top_idxs = np.argpartition(np.reshape(sqdiff, -1), top_k)[:top_k]
    min_idxs = [np.unravel_index(ii, (emb1.shape[0], emb2.shape[0])) for ii in top_idxs]
    min_dists = [sqdiff[ii] for ii in min_idxs]
    
    return np.min(sqdiff), min_idx, min_dists, min_idxs
    
