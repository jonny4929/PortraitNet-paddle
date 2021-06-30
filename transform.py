import numpy
import cv2
from numpy.lib.twodim_base import mask_indices

class RandomHorizontalFlip():
    def __init__(self,prob=0.5):
        self.prob=prob
    
    def __call__(self,img,mask):
        if numpy.random.uniform(0,1)<self.prob:
            img=cv2.flip(img,1)
            mask=cv2.flip(mask,1)
        return img,mask

class RandomRotation():
    def __init__(self,max_rotation=45,resize_range=(0.5,1.5)):
        self.max_rotation=max_rotation
        self.resize_range=resize_range
    
    def __call__(self, img, mask):
        h,w=img.shape[:2]
        rot=numpy.random.uniform(-self.max_rotation,self.max_rotation)
        resize=numpy.random.uniform(*self.resize_range)
        M=cv2.getRotationMatrix2D((h//2,w//2), rot, resize)
        img=cv2.warpAffine(img,M,(w,h))
        mask=cv2.warpAffine(mask,M,(w,h))
        return img,mask

class Resize():
    def __init__(self,shape):
        self.shape=shape

    def __call__(self,img,mask):
        img=cv2.resize(img,self.shape)
        mask=cv2.resize(mask,self.shape,interpolation=cv2.INTER_NEAREST)
        return img,mask


class RandomHSV():
    def __init__(self,
                hue_prob=0.5,hue_range=(0.4,1.7),
                saturation_prob=0.5,saturation_range=(0.4,1.7),
                value_prob=0.5,value_range=(0.4,1.7)
                ):
        self.hue_prob=hue_prob
        self.hue_range=hue_range
        self.saturation_prob=saturation_prob
        self.saturation_range=saturation_range
        self.value_prob=value_prob
        self.value_range=value_range

    def __call__(self, img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        if numpy.random.uniform()<self.hue_prob:
            h=numpy.random.uniform(self.hue_range[0],self.hue_range[1])
            img[:,:,0]=img[:,:,0]*h
            img[:,:,0]=numpy.clip(img[:,:,0],0,179)
        if numpy.random.uniform()<self.saturation_prob:
            s=numpy.random.uniform(self.saturation_range[0],self.saturation_range[1])
            img[:,:,1]=img[:,:,1]*s
        if numpy.random.uniform()<self.value_prob:
            v=numpy.random.uniform(self.value_range[0],self.value_range[1])
            img[:,:,2]=img[:,:,2]*v
        img=numpy.clip(img,0,255)
        img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
        return img

class RandomContrast():
    def __init__(self,prob=0.5,randrange=(0.6,1.5)):
        self.prob=prob
        self.range=randrange

    def __call__(self,img):
        if numpy.random.uniform()<self.prob:
            c=numpy.random.uniform(*self.range)
            for i in range(3):
                maxc,minc=numpy.max(img[:,:,i]),numpy.min(img[:,:,i])
                img[:,:,i]=(img[:,:,i]-minc)*c+(128-(maxc-minc)*c/2)
            img=numpy.clip(img,0,255)
        return img

class RandomTranslate():
    def __init__(self,randrange=0.25):
        self.range=randrange

    def __call__(self,img,label):
        c1=numpy.random.uniform(-self.range,self.range)
        c2=numpy.random.uniform(-self.range,self.range)
        M = numpy.array([[1,0,c1*img.shape[1]],[0,1,c2*img.shape[0]]])
        img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
        label = cv2.warpAffine(label,M,(label.shape[1],label.shape[0]))
        return img,label

class RandomBlur():
    def __init__(self,prob=0.5):
        self.prob=prob

    def __call__(self,img):
        if numpy.random.uniform()<self.prob:
            return img
        ks=numpy.random.randint(3, 10)
        img = cv2.blur(img, (ks, ks))
        return img

class RandomNoise():
    def __init__(self,prob=0.5):
        self.prob=prob

    def __call__(self,img):
        if numpy.random.uniform()<self.prob:
            return img
        img = cv2.GaussianBlur(img, (0,0), 10)
        return img
