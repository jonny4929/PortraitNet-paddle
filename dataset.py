import os
import numpy as np
import paddle
import random
import cv2
from scipy.ndimage.morphology import distance_transform_edt
from transform import *

class dataset(paddle.io.Dataset):
    def __init__(self,path,data_root, img_shape=(224,224),edge=True,kl=True, aug=True):
        f=open(path).readlines()
        f=list(set(f))
        self.data_list=[]
        for line in f:
            img,mask=line.split()
            img=os.path.join(data_root,img)
            mask=os.path.join(data_root,mask)
            self.data_list.append((img,mask))
        self.hflip=RandomHorizontalFlip()
        self.rotation=RandomRotation()
        self.resize=Resize(img_shape)
        self.translate=RandomTranslate()
        self.randhsv=RandomHSV()
        self.randc=RandomContrast()
        self.blur=RandomBlur()
        self.gaussian=RandomNoise()
        self.norm=paddle.vision.transforms.Compose([paddle.vision.transforms.Normalize()])
        self.edge=edge
        self.kl=kl
        self.aug=aug

    def __getitem__(self,idx):
        img,mask=self.data_list[idx]
        img=cv2.imread(img,1)
        if img is None:
            print(img)
        mask=cv2.imread(mask,0)
        mask[mask>1]=1
        if self.aug is False:
            img,mask=self.resize(img,mask)
            img=self.norm(img.transpose([2,0,1]))
            return img,img,mask,mask
        img,mask=self.resize(*self.rotation(*self.hflip(img,mask)))
        if self.kl is True:
            img_aug=self.randc(self.randhsv(img))
            img=self.norm(img.transpose([2,0,1]))
            img_aug=self.norm(img_aug.transpose([2,0,1]))
        if self.edge is True:
            edge=mask_to_binary_edge(mask,4,2)
        return img,img_aug,mask.astype('int64'),edge.astype('int64')

    def __len__(self):
        return len(self.data_list)

#the following part is copied from paddleseg
#
def mask_to_onehot(mask, num_classes):
    """
    Convert a mask (H, W) to onehot (K, H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Onehot mask with shape(K, H, W).
    """
    _mask = [mask == i for i in range(num_classes)]
    _mask = np.array(_mask).astype(np.uint8)
    return _mask


def onehot_to_binary_edge(mask, radius):
    """
    Convert a onehot mask (K, H, W) to a edge mask.

    Args:
        mask (np.ndarray): Onehot mask with shape (K, H, W)
        radius (int|float): Radius of edge.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    if radius < 1:
        raise ValueError('`radius` should be greater than or equal to 1')
    num_classes = mask.shape[0]

    edge = np.zeros(mask.shape[1:])
    # pad borders
    mask = np.pad(
        mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(
            mask[i, :]) + distance_transform_edt(1.0 - mask[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edge += dist

    edge = np.expand_dims(edge, axis=0)
    edge = (edge > 0).astype(np.uint8)
    return edge


def mask_to_binary_edge(mask, radius, num_classes):
    """
    Convert a segmentic segmentation mask (H, W) to a binary edge mask(H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        radius (int|float): Radius of edge.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    mask = mask.squeeze()
    onehot = mask_to_onehot(mask, num_classes)
    edge = onehot_to_binary_edge(onehot, radius)
    return edge


