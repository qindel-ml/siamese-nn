from PIL import Image
from utils import *
class Letterbox(object):
    """
    A minimal letterbox class.
    """
    
    def __init__(self, img):
        self._img = img.copy()
        self.transforms_ = [] # the log of the image transformations
        self.fill_color_ = (0, 0, 0)
        
    def __getattr__(self, key):
        if key == '_img':
            raise AttributeError()
        return getattr(self._img, key)
    
    def letterbox(self, sizew=224,sizeh=224, augments=None, randomize_pos=False, fill_letterbox=False):
        """
        The letterboxing routine. It is assumed that the geometric augmentation have been performed.
        
        Args:
            sizew, sizeh: the target image sizes
            augments: the image augmentation parameters (ignored in this class)
            randomize_pos: randomize the position within letterbox (if possible)
            fill_letterbox: enlarge small images to fill the letterbox
        """

        # compute the new image scale
        scale = min([sizew / self._img.width, sizeh / self._img.height])
        
        # rescale the image if it is too large
        if scale < 1:
            self._img = self._img.resize((int(scale * self._img.width), int(scale * self._img.height)), Image.BILINEAR)
            scale_ = scale
            
        # rescake the image if it is too small and the letterbox should be filled
        if fill_letterbox and scale > 1:
            self._img = self._img.resize((int(scale * self._img.width), int(scale * self._img.height)), Image.NEAREST)
            scale_ = scale
        else:
            scale_ = 1
             
        # compute the image position
        if randomize_pos:
            dx = np.random.randint(0, sizew - self._img.width + 1)
            dy = np.random.randint(0, sizeh - self._img.height + 1)
        else:
            dx = (sizew - self._img.width) // 2
            dy = (sizeh - self._img.height) // 2
                        
        # append the final transformation
        self.transforms_.append({'scale_shift':[scale_, scale_, dx, dy]})
        
        # paste the rescaled image
        new_img = Image.new('RGB', (sizew, sizeh), self.fill_color_)
        new_img.paste(self._img, (dx, dy))
        self._img = new_img
	
