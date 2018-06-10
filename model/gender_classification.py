#coding:utf-8
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py 
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from read_img import read_img
from CNN_model import CNN_model_gender



path='/Users/angle_yao/简历项目/gender_classification/性别识别训练样本/'
w=92
h=112
c=3
n_classes =2
input_shape=(w,h,c)

#导入训练集和测试集
data,label=read_img(path)
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#将所有数据分为训练集和测试集
ratio=0.8
s=np.int(num_example*ratio)

#训练集
train_x=data[:s]
train_y=label[:s]
#one-hot编码
train_y=to_categorical(train_y)

#测试集
test_x=data[s:]
test_y=label[s:]
#one-hot编码
test_y=to_categorical(test_y)

train_x=np.reshape(train_x,[train_x.shape[0],w,h,c])
test_x=np.reshape(test_x,[test_x.shape[0],w,h,c])
#print(train_y.shape)

model_ng=CNN_model_gender(input_shape,n_classes)
init=tf.initialize_all_variables()

with tf.Session() as sess: 
	 
	 sess.run(init)
	 filepath="CNN_model_gender.h5"
	 checkpoint=ModelCheckpoint(filepath,monitor='val_acc',save_best_only=True,save_weights_only=False,period=1,mode='max')
	 callbacks_list = [checkpoint]
	 model_ng.fit(train_x,train_y,validation_data=(test_x,test_y), batch_size=32,epochs=20)
	  
	 model_ng=load_model('CNN_model_gender.h5')
	 print(model_ng.evaluate(test_x,test_y)[1])
	 y_pred=model_ng.predict(test_x)
	 C=confusion_matrix(test_y.argmax(axis=1),y_pred.argmax(axis=1))
	 print(C)


	
	 