"""
Loads the data from a data directory.
"""
import json
import os


def load_data(data_dir, allowed_sections=None, verbose=True):
    """
    Args:
        data_dir: the images data directory
        allowed_sections: a list of allowed sections (integers 1, 2 or 3). Set to None to disable
    Returns:
        data: a list of dictionaries describing the images
    """
    # get a list of all JSON files in the directory
    files = []
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        files.extend(filenames)
    files = [x for x in files if x.endswith('.json')]

    # join all lines from each file (each line is a dictionary)
    data = []
    for file in files:
        with open(os.path.join(data_dir, file), 'r') as inpf:
            if verbose:
                print('Loading from {}'.format(os.path.join(data_dir, file)))
            data.extend([json.loads(x) for x in inpf.readlines()])
    if not(allowed_sections is None):
        data = [x for x in data if x['section'] in allowed_sections]
    return data
