# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import stats
from scipy.ndimage.filters import convolve

data_dir = 'C:/MRI/MS-LAQ/'

malf_classes = ['bg', 'bs', 'cgm', 'crblr_gm', 'crblr_wm', 'csf', 'dgm', 'lv', 'ov', 'wm']
modalities = ['t1p', 't2w', 'pdw', 'flr']
good_malf_classes = ['cgm', 'dgm', 'wm']

class mri:
    t1p = ''
    lesions = ''
    
    malf = {}
    
    folder = ''
    
    priors = {}    
    def __init__(self, t1p_image):
        
        tokens = t1p_image.split('_')
        
        self.folder = data_dir + tokens[2] + '_' + tokens[3] + '/m0/'
        
        self.priors['t1p'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t1p_ISPC-stx152lsq6.mnc.gz'
        self.priors['t2w'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t2w_ISPC-stx152lsq6.mnc.gz'
        self.priors['pdw'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_pdw_ISPC-stx152lsq6.mnc.gz'
        self.priors['flr'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_flr_ISPC-stx152lsq6.mnc.gz'          
        
        self.lesions = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_ct2f_ISPC-stx152lsq6.mnc.gz'

        for tissue in malf_classes:
            self.malf[tissue] = self.folder + 'malf/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_prior_' + tissue + '_ISPC-stx152lsq6.mnc.gz'

mri_list = []


for root, dirs, filenames in os.walk(data_dir):        
    for f in filenames:
        if f.endswith('_m0_t1p.mnc.gz'):
            mri_list.append(mri(f))

#malf_thresh = {}
malf_tissues = {}

scales = [1,2,3,4,5]

feature_map = np.zeros((np.shape(mri_list[0].priors['t1p'])[0], np.shape(mri_list[0].priors['t1p'])[1], np.shape(mri_list[0].priors['t1p'][2]), len(scales)))

for img in mri_list:
    print img.priors['t1p']

    tissues = {}
    for mod in modalities:
        tissues[mod] = nib.load(img.priors[mod]).get_data()

    for m in malf_classes:
        malf_tissues[m] = nib.load(img.malf[m]).get_data()
#        malf_thresh[m] = np.greater_equal(malf_tissues[m], 0.7)
        for s in scales:
            feature_map = convolve(malf_tissues[m],np.ones((s,s,s)))

print 'done'