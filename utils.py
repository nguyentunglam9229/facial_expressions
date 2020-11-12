#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import h5py
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[33]:


def read_images():
    """
    read images in images folder
    return: images, filenames as numpy array 
    """
    files = os.listdir('images/')
    imageList = []
    fileNames = []
    for file in files:
        image = cv2.imread(f"images/{file}", 0)
        if image.shape == (350, 350):
            imageList.append(image)
            fileNames.append(file)
    return np.array(imageList), np.array(fileNames)


# In[34]:


def read_legend():
    """
    read legend of images from data/lenged.csv
    return lengend data, list of target values
    """
    df = pd.read_csv('data/legend.csv')
    df['emotion'] = df['emotion'].str.lower()
    label_encoder = LabelEncoder()
    emotion = df['emotion']
    integer_encoded = label_encoder.fit_transform(emotion)
    df['emotion_encoded'] = integer_encoded
    
    return df, label_encoder.inverse_transform([0,1,2,3,4,5,6,7])


# In[35]:


def delete_unmatched_image(imgs, fileNames, df): 
    y = []
    for file in fileNames:
        value = df[df['image'] == file].emotion_encoded.values
        if len(value) == 1:
            y.append(value[0])
        else:
            y.append(-1)
    y_tmp = np.array(y)
    deleteList = np.where(y_tmp == -1)[0]
    images = np.delete(imgs, deleteList, 0)
    y_target = np.delete(y_tmp, deleteList)
    file_names = np.delete(fileNames, deleteList)
    
    return images, y_target, file_names


# In[36]:


def pre_process():
    imgs, fileNames = read_images()
    df, target_names = read_legend()
    images, y_target, file_names = delete_unmatched_image(imgs, fileNames, df)
    y_target=pd.get_dummies(y_target)
    
    return images, y_target, target_names, file_names
    

# In[17]:


def store_many_hdf5(images, target):
    file = h5py.File('h5/images6.h5', 'w')
    dataset = file.create_dataset('images', np.shape(images), data=images)
    output = file.create_dataset('target', np.shape(target), data=target)
    file.close()


