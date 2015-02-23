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

f = h5py.File(data_dir + 'features2.hdf5', 'r')

#features types: malf-context*scales + image-intensities
#features types: 10*5 + 4

image_pixels = 60*256*256

priors = 10
feature_scales = 2
num_mods = 4

num_train = 15
num_test = 5




training_vector = np.zeros(shape=(num_train, image_pixels*(priors*feature_scales+num_mods)))
print 'train_vector', np.shape(training_vector)

for i, img in enumerate(mri_list[0:num_train]):
    print i
    features = f[img.uid]
    flat_feature = np.reshape(features, image_pixels*feature_scales*priors)
    #print 'flat_features', np.shape(flat_feature)
    training_vector[i, 0:image_pixels*(feature_scales*priors)] = flat_feature
    #print 'begin ', 0:60*256*256*50
    
    for j, m in enumerate(modalities):
        malf_data = nib.load(img.images[m]).get_data()
        flat_malf_data = np.reshape(malf_data, (image_pixels))
        #print 'flat_malf', np.shape(flat_malf_data)
        #print 'train_vector_slice', np.shape(training_vector[i, 60*256*256*(50+j):60*256*256*(50+j+1)])
        
        #print 'be gin: ', 60*256*256*(50+j+1)
        #print 'end: ', 60*256*256*(50+j+2)
        
        training_vector[i, image_pixels*(feature_scales*priors+j):image_pixels*(feature_scales*priors+j+1)] = flat_malf_data
    
test_vector = np.zeros(shape=(5, image_pixels*(feature_scales*priors + num_mods)))
for i, img in enumerate(mri_list[num_train:num_train+num_test]):
    features = f[img.uid]
    test_vector[i, 0:image_pixels*(feature_scales*priors)] = np.reshape(features, image_pixels*feature_scales*priors)
    
    for j, m in enumerate(modalities):
        malf_data = nib.load(img.images[m]).get_data()
        flat_malf_data = np.reshape(malf_data, (image_pixels))
        test_vector[i, image_pixels*(feature_scales*priors+j):image_pixels*(feature_scales*priors+j+1)] = flat_malf_data


train_labels = np.zeros(shape=(num_train, image_pixels))
for i, img in enumerate(mri_list[0:num_train]):
    train_labels[i, :] = np.reshape(nib.load(img.lesions).get_data(), image_pixels)

test_labels = np.zeros(shape=(num_test, image_pixels))
for i, img in enumerate(mri_list[num_train:num_train+num_test]):
    test_labels[i, :] = np.reshape(nib.load(img.lesions).get_data(), image_pixels)



forest = RandomForestClassifier()
forest.fit(training_vector, train_labels)
forest.predict(test_vector)