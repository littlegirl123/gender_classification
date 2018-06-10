# coding=utf-8
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from skimage import io,transform
import tensorflow as tf
import glob
import numpy as np 

w=92
h=112
c=3

def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.bmp'):
            #print('reading the images: %s' %(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

