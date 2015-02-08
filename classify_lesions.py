# -*- coding: utf-8 -*-
"""
Created on Thu Feb 05 19:59:43 2015

@author: Andrew
"""
import nibabel as nib
import numpy as np
import os
import h5py

from mri import mri
from sklearn.ensemble import RandomForestClassifier

data_dir = 'C:/MRI/MS-LAQ/'

malf_classes = ['bg', 'bs', 'cgm', 'crblr_gm', 'crblr_wm', 'csf', 'dgm', 'lv', 'ov', 'wm']
modalities = ['t1p', 't2w', 'pdw', 'flr']
good_malf_classes = ['cgm', 'dgm', 'wm']


mri_list = []

for root, dirs, filenames in os.walk(data_dir):        
    for f in filenames:
        if f.endswith('_m0_t1p.mnc.gz'):
            mri_list.append(mri(f))

f = h5py.File(data_dir + 'features.hdf5', 'r')

#features types: malf-context*scales + image-intensities
#features types: 10*5 + 4
training_vector = np.zeros(shape=(15, 60*256*256*54))
print 'train_vector', np.shape(training_vector)

for i, img in enumerate(mri_list[0:15]):
    features = f[img.uid]
    flat_feature = np.reshape(features, 60*256*256*50)
    #print 'flat_features', np.shape(flat_feature)
    training_vector[i, 0:60*256*256*50] = flat_feature
    #print 'begin ', 0:60*256*256*50
    
    for j, m in enumerate(modalities):
        malf_data = nib.load(img.images[m]).get_data()
        flat_malf_data = np.reshape(malf_data, (60*256*256))
        #print 'flat_malf', np.shape(flat_malf_data)
        #print 'train_vector_slice', np.shape(training_vector[i, 60*256*256*(50+j):60*256*256*(50+j+1)])
        
        #print 'be gin: ', 60*256*256*(50+j+1)
        #print 'end: ', 60*256*256*(50+j+2)
        
        training_vector[i, 60*256*256*(50+j):60*256*256*(50+j+1)] = flat_malf_data
    
test_vector = np.zeros(shape=(5, 60*256*256*54))
for i, img in enumerate(mri_list[15:20]):
    features = f[img.uid]
    test_vector[i, 0:60*256*256*50] = np.reshape(features, 60*256*256*50)
    
    for j, m in enumerate(modalities):
        malf_data = nib.load(img.images[m]).get_data()
        flat_malf_data = np.reshape(malf_data, (60*256*256))
        test_vector[i, 60*256*256*(50+j):60*256*256*(50+j+1)] = flat_malf_data


train_labels = np.zeros(shape=(15, 60*256*256))
for i, img in enumerate(mri_list[0:15]):
    train_labels[i, :] = nib.load(img.lesions).get_data()

test_labels = np.zeros(shape=(5, 60*256*256))
for i, img in enumerate(mri_list[15:20]):
    test_labels[i, :] = nib.load(img.lesions).get_data()



forest = RandomForestClassifier()
forest.fit(training_vector, train_labels)
forest.predict(test_vector)