# -*- coding: utf-8 -*-
"""
Created on Thu Feb 05 19:59:43 2015

@author: Andrew
"""
import nibabel as nib
import numpy as np
import os
import h5py

from scipy.ndimage.filters import gaussian_filter
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier



data_dir = 'G:/MRI/MS-LAQ/'

malf_classes = ['bg', 'bs', 'cgm', 'crblr_gm', 'crblr_wm', 'csf', 'dgm', 'lv', 'ov', 'wm']
modalities = ['t1p', 't2w', 'pdw', 'flr']
good_malf_classes = ['cgm', 'dgm', 'wm']

class mri:
    t1p = ''
    lesions = ''
    
    priors = {}
    
    folder = ''
    uid = ''
    
    images = {}    
    def __init__(self, t1p_image):
        
        tokens = t1p_image.split('_')
        
        self.folder = data_dir + tokens[2] + '_' + tokens[3] + '/m0/'
        
        self.uid = tokens[2] + tokens[3]        
        
        self.images['t1p'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t1p_ISPC-stx152lsq6.mnc.gz'
        self.images['t2w'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t2w_ISPC-stx152lsq6.mnc.gz'
        self.images['pdw'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_pdw_ISPC-stx152lsq6.mnc.gz'
        self.images['flr'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_flr_ISPC-stx152lsq6.mnc.gz'          
        
        self.lesions = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_ct2f_ISPC-stx152lsq6.mnc.gz'

        for tissue in malf_classes:
            self.priors[tissue] = self.folder + 'malf/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_prior_' + tissue + '_ISPC-stx152lsq6.mnc.gz'


mri_list = []

for root, dirs, filenames in os.walk(data_dir):        
    for f in filenames:
        if f.endswith('_m0_t1p.mnc.gz'):
            mri_list.append(mri(f))

#features types: malf-context*scales + image-intensities
#features types: 10*5 + 4

image_pixels = 60*256*256

scales = [1]
priors = len(malf_classes)
feature_scales = len(scales)
num_mods = len(modalities)

num_train = 5
num_test = 5

features = []
labels = []


for i, img in enumerate(mri_list[0:num_train+num_test]):
    print i, '/', num_train+num_test
    not_background = np.nonzero(nib.load(img.priors['bg']).get_data() < 0.3)
    feature = np.zeros((np.shape(not_background)[1], num_mods + priors*feature_scales))
        
    for j, malf in enumerate(malf_classes):
        malf_image = nib.load(img.priors[malf]).get_data()
        
        for k, s in enumerate(scales):
            filtered = gaussian_filter(malf_image, s)
            feature[:, j*feature_scales + k] = filtered[not_background]

    for j, mod in enumerate(modalities):
        feature[:, priors*feature_scales + j] = nib.load(img.images[mod]).get_data()[not_background]


    for f in feature:
        features.append(f)

    image_labels = nib.load(img.lesions).get_data()[not_background]
    
    for l in image_labels:
        labels.append(l)


training_features = features[0:np.shape(features)[0]/2]
test_features = features[np.shape(features)[0]/2:]
training_labels = labels[0:len(labels)/2]
test_labels = labels[len(features)/2:]

print "done getting features & labels"


print np.shape(training_features)

for g in [1,2,3,4,5]:
    
    #mix_model = GMM(n_components=g)
    #mix_model.fit(training_features)
    #predictions = mix_model.predict(test_features)
    

    forest = RandomForestClassifier(n_estimators=g*100)    
    forest.fit(training_features, training_labels)
    forest.predict(test_features)
    
    
    print "done predictions at level", g
    
    tp = 0
    fp = 0
    fn = 0
    
    total_voxels = image_pixels*num_test
        
    for i, p in enumerate(predictions):        
        if p > 0.0 and test_labels[i] > 0.0:
            tp+=1
        if p > 0.0 and test_labels[i] == 0.0:
            fp+=1
        if p == 0.0 and test_labels[i] > 0.0:
            fn+=1
    tp = 0
    fp = 0
    fn = 0
            
    print "true positives: ", tp
    print "false positives: ", fp
    print "false negatives: ", fn