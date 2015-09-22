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
from sklearn.naive_bayes import GaussianNB

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

num_train = 5
num_test = 5

training_vector = np.zeros((num_train*image_pixels, priors*feature_scales + num_mods))
print "training vector size:", np.shape(training_vector)

for i, img in enumerate(mri_list[0:num_train]):
    print i
    features = f[img.uid]
    
    training_vector[(i)*image_pixels:(i+1)*image_pixels, 0:priors*feature_scales] = np.reshape(features, (image_pixels, feature_scales*priors))

    for j, mod in enumerate(modalities):
        image_data = nib.load(img.images[mod]).get_data()
        training_vector[(i)*image_pixels:(i+1)*image_pixels, priors*feature_scales + j] = np.reshape(image_data, image_pixels)


test_vector = np.zeros(shape=(num_test*image_pixels, feature_scales*priors + num_mods))

for i, img in enumerate(mri_list[num_train:num_train+num_test]):
    features = f[img.uid]
    test_vector[(i)*image_pixels:(i+1)*image_pixels, 0:priors*feature_scales] = np.reshape(features, (image_pixels, feature_scales*priors))
    
    for j, mod in enumerate(modalities):
        image_data = nib.load(img.images[mod]).get_data()
        test_vector[(i)*image_pixels:(i+1)*image_pixels, priors*feature_scales + j] = np.reshape(image_data, image_pixels)

print "loading lesion labels..."

train_labels = np.zeros(shape=(num_train*image_pixels))
for i, img in enumerate(mri_list[0:num_train]):
    train_labels[(i)*image_pixels:(i+1)*image_pixels] = np.reshape(nib.load(img.lesions).get_data(), image_pixels)

test_labels = np.zeros(shape=(num_test*image_pixels))
for i, img in enumerate(mri_list[num_train:num_train+num_test]):
    test_labels[(i)*image_pixels:(i+1)*image_pixels] = np.reshape(nib.load(img.lesions).get_data(), image_pixels)

print "training classifier..."

gnb = GaussianNB()
gnb.fit(training_vector, train_labels)
predictions = gnb.predict(test_vector)

print "done predictions"

tp = 0
fp = 0
fn = 0

total_voxels = image_pixels*num_test

print np.shape(predictions), np.shape(test_labels)

for i, p in enumerate(predictions):
    #print p, test_labels[i]    
    
    if p > 0.0 and test_labels[i] > 0.0:
        tp+=1
    if p > 0.0 and test_labels[i] == 0.0:
        fp+=1
    if p == 0.0 and test_labels[i] > 0.0:
        fn+=1
        
print "true positives: ", tp
print "false positives: ", fp
print "false negatives: ", fn
print "total lesions: ", np.count_nonzero(test_labels)