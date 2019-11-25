import numpy as np
def float_rand(a=0, b=1, size=None):
    """
    Returns a floating random number uniformly sampled from [a, b).
    
    Args:
        a, b: the interval limits
        size: the return array size (use None for a single value)
    """
    
    return np.random.random_sample(size) * (b - a) + a

def int_rand(a=0, b=0, size=None):
    """
    Returns an integer random number uniformly sampled from [a, b].
    
    Args:
        a, b: the interval limits
        size: the return array size (use None for a single value)
    """
    
    return np.random.randint(a, b + 1, size)

def prob_choice(p):
    """
    Return True if a random number drawn from [0, 1) is less than p.
    
    Args:
        p: the probability
    """
    
    return np.random.random_sample() < p
