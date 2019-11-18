from PIL import Image
import numpy as np
import cv2
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


class LetterboxImage(object):
    def __init__(self, img):
        self._img = img.copy()
        self.colors_ = 'L' if len(np.array(self._img).shape)==2 else 'RGB'

    def __getattr__(self, key):
        if key == '_img':
            raise AttributeError()
        return getattr(self._img, key)


    def do_letterbox(self, sizew, sizeh, randomize_pos=True):

        curw, curh = self._img.size

        # resize image if necessary
        new_scale=min([sizew / curw, sizeh / curh])
        target_w = int(curw * new_scale)
        target_h = int(curh * new_scale)
        
        if new_scale < 1:
            res = cv2.resize(np.asarray(self._img), dsize=(target_w, target_h), interpolation=cv2.INTER_LINEAR)
            self._img = Image.fromarray(res)
            curw, curh = self._img.size
        
        # create the final image and fill it
        if self.colors_ == 'L':
            fill_color = 128
        else:
            fill_color = (128, 128, 128)
        new_img = Image.new(self.colors_, (sizew, sizeh), fill_color)

        if randomize_pos:
            dx = np.random.randint(low=0, high=max([0, sizew - curw])+1)
        else:
            dx = max([0, (sizew - curw)//2])
 
        if randomize_pos:
            dy = np.random.randint(low=0, high=max([0, sizeh - curh])+1)
        else:
            dy = max([0, (sizeh - curh)//2])

            
        new_img.paste(self._img, (dx, dy))
        self._img = new_img
        curw, curh = self._img.size

    
    def do_augment(self, augment={}):
        """
        Augment an image
        """
        iw, ih = self._img.size
       
        curw = iw
        curh = ih
        
        # augment?
        if not augment:
            do_augment = False
            return
        else:
            do_augment = True
            
        # crop the original image
        if do_augment:
            if 'crop_prob' in augment:
                if np.random.random() < augment['crop_prob']:
                    area_frac = 1 - np.random.random() * augment['crop_frac']
                    neww = int(np.sqrt(area_frac) * curw)
                    newh = int(np.sqrt(area_frac) * curh)
                    dx = np.random.randint(low = 0, high = max([0, curw - neww]) + 1)
                    dy = np.random.randint(low = 0, high = max([0, curh - newh]) + 1)
                    crop_img = self._img.crop((dx, dy, neww, newh))
                    self._img = crop_img
                    curw, curh = self._img.size
                    
            if 'jitter_prob' in augment:
                if np.random.random() < augment['jitter_prob']:
                    old_ar = curw / curh
                    ar_fact = 1 + (np.random.random() * 2. - 1) * augment['jitter']
                    if np.random.random() < 0.5:
                        ar_fact = 1. / ar_fact
                        
                    new_ar = old_ar * ar_fact
                    curw = int(curh * new_ar)
            
        new_scale = 1.0
    
        target_w = int(curw * new_scale)
        target_h = int(curh * new_scale)
        new_scale=min([target_w / iw, target_h / ih])

        if new_scale<1:
            res = cv2.resize(np.asarray(self._img), dsize=(target_w, target_h), interpolation=cv2.INTER_LINEAR)
            self._img = Image.fromarray(res)
        if new_scale>=1:
            res = cv2.resize(np.asarray(self._img), dsize=(target_w, target_h), interpolation=cv2.INTER_NEAREST)
            self._img = Image.fromarray(res)
        curw, curh = self._img.size

        if do_augment:
            if 'rot' in augment:
                rot = np.random.random() < augment['rot']
            else:
                rot = False
            
            if 'hflip' in augment:
                hflip = np.random.random() < augment['hflip']
            else:
                hflip = False

            if 'vflip' in augment:
                vflip = np.random.random() < augment['vflip']
            else:
                vflip = False
            
            if rot:
                self._img = self._img.transpose(Image.ROTATE_90)
                curw, curh = self._img.size
                
            if hflip:
                self._img = self._img.transpose(Image.FLIP_LEFT_RIGHT)
                
            if vflip:
                self._img = self._img.transpose(Image.FLIP_TOP_BOTTOM)

            dist_col = False
            if self.colors_ == 'RGB':
                if 'hue' in augment:
                    self.hue = flrand(-augment['hue'], augment['hue'])
                    dist_col = True
                    self.aug_hue = self.hue

                if 'sat' in augment:
                    self.sat = flrand(1, 1 + augment['sat']) if flrand() < 0.5 else 1 / flrand(1, 1 + augment['sat'])
                    dist_col = True
                    self.aug_sat = self.sat

                if 'val' in augment:
                    self.val = flrand(1, 1 + augment['val']) if flrand() < 0.5 else 1 / flrand(1, 1 + augment['val'])
                    dist_col = True
                    self.aug_val = self.val

                if dist_col:
                    x = rgb_to_hsv(np.array(self._img)/255.)
                    x[..., 0] += self.hue
                    x[..., 0][x[..., 0]>1] -= 1
                    x[..., 0][x[..., 0]<0] += 1

                    x[..., 1] *= self.sat
                    x[..., 2] *= self.val
                    x[x>1] = 1
                    x[x<0] = 0

                    self._img = Image.fromarray(np.uint8(hsv_to_rgb(x) * 255))

        
        
def flrand(a=0, b=1):
    return (b - a) * np.random.rand() + a
