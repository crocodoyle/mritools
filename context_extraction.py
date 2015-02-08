# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from mri import mri
from scipy import stats
from scipy.ndimage.filters import convolve

import h5py


data_dir = 'C:/MRI/MS-LAQ/'

malf_classes = ['bg', 'bs', 'cgm', 'crblr_gm', 'crblr_wm', 'csf', 'dgm', 'lv', 'ov', 'wm']
modalities = ['t1p', 't2w', 'pdw', 'flr']
good_malf_classes = ['cgm', 'dgm', 'wm']


mri_list = []


for root, dirs, filenames in os.walk(data_dir):        
    for f in filenames:
        if f.endswith('_m0_t1p.mnc.gz'):
            mri_list.append(mri(f))

#malf_thresh = {}
malf_tissues = {}

scales = [1,2,3,4,5]


size = nib.load(mri_list[0].priors['t1p'])


feature_map = np.zeros(shape=(np.shape(size)[0], np.shape(size)[1], np.shape(size)[2], len(malf_classes), len(scales)))


f = h5py.File(data_dir + "features2.hdf5", "w")
for i, img in enumerate(mri_list):
    print i, '/', len(mri_list), img.uid

    tissues = {}
    for mod in modalities:
        tissues[mod] = nib.load(img.priors[mod]).get_data()

    for j, m in enumerate(malf_classes):
        malf_tissues[m] = nib.load(img.malf[m]).get_data()
#        malf_thresh[m] = np.greater_equal(malf_tissues[m], 0.7)
        for s in scales:
            feature_map[:,:,:,j,s-1] = convolve(malf_tissues[m],np.ones((s,s,s)))
    
    dset = f.create_dataset(img.uid, np.shape(feature_map), dtype='f', compression="gzip")
    dset[...] = feature_map

f.close()

print 'done'