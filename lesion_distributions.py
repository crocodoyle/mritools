# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import theano
import pylearn2

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import stats

data_dir = 'C:/MRI/MS-LAQ/'

malf_classes = ['bg', 'bs', 'cgm', 'crblr_gm', 'crblr_wm', 'csf', 'dgm', 'lv', 'ov', 'wm']


class mri:
    t1p = ''
    lesions = ''
    
    malf = {}
    
    folder = ''
    
    def __init__(self, t1p_image):
        
        tokens = t1p_image.split('_')
        
        self.folder = data_dir + tokens[2] + '_' + tokens[3] + '/m0/'
        
        self.t1p = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t1p_ISPC-stx152lsq6.mnc.gz'
        self.t2w = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t2w_ISPC-stx152lsq6.mnc.gz'
        self.pdw = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_pdw_ISPC-stx152lsq6.mnc.gz'
        self.flr = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_flr_ISPC-stx152lsq6.mnc.gz'          
        
        self.lesions = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_ct2f_ISPC-stx152lsq6.mnc.gz'

        for tissue in malf_classes:
            self.malf[tissue] = self.folder + 'malf/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_prior_' + tissue + '_ISPC-stx152lsq6.mnc.gz'

mri_list = []


for root, dirs, filenames in os.walk(data_dir):        
    for f in filenames:
        if f.endswith('_m0_t1p.mnc.gz'):
            mri_list.append(mri(f))


malf_lesions = {}
malf_thresh = {}
malf_tissues = {}
malf_lesion_locations = {}

for m in malf_classes:
    malf_lesions[m] = []
    
for img in mri_list:    
    lesions = nib.load(img.lesions).get_data()

    t1p = nib.load(img.t1p).get_data()
    
    for m in malf_classes:
        malf_tissues[m] = nib.load(img.malf[m]).get_data()
        malf_thresh[m] = np.greater_equal(malf_tissues[m], 0.7)
        malf_lesion_locations[m] = np.multiply(lesions, malf_thresh[m])


    for m in malf_classes:
        for lesion_voxel in t1p[np.nonzero(malf_lesion_locations[m])]:
            malf_lesions[m].append(lesion_voxel)


for m in malf_classes:
    try:
        print m
        
        kde = stats.gaussian_kde(malf_lesions[m])
        X_plot = np.linspace(0, 256, 1000)
        density = kde(X_plot)

        plt.plot(X_plot, density)
        plt.title(m + ', ' + str(np.shape(malf_lesions[m])[0]) + ' voxels')
        plt.xlabel('voxel intensity')
        plt.ylabel('proportion of voxels')
        plt.savefig(data_dir + m +'.jpg')
        plt.show()
    except Exception as e:
        print 'couldnt do {0}'.format(m)
        print e.message
#plt.close()

print 'done'