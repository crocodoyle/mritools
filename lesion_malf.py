# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:50:06 2015

@author: Andrew
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from mri import mri

import h5py


data_dir = 'C:/MRI/MS-LAQ/'

malf_classes = ['bg', 'bs', 'cgm', 'crblr_gm', 'crblr_wm', 'csf', 'dgm', 'lv', 'ov', 'wm']
modalities = ['t1p', 't2w', 'pdw', 'flr']


mri_list = []


for root, dirs, filenames in os.walk(data_dir):        
    for f in filenames:
        if f.endswith('_m0_t1p.mnc.gz'):
            mri_list.append(mri(f))

malf_tissues = {}

size = nib.load(mri_list[0].priors['cgm'])