# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 16:18:40 2015

@author: Andrew
"""

import nibabel as nib
import numpy as np
import os

data_dir = 'C:/MRI/MS-LAQ/'


mri_list = []

for root, dirs, filenames in os.walk(data_dir):
    for name in filenames:
        if '.mnc.gz' in name:
            try:
                input_img = nib.load(os.path.join(root, name))
                tokens = os.path.join(root, name).split('.')
                output_name = tokens[0] + '.nii.gz'
                print output_name
                nib.save(input_img, output_name)
            except Exception as e:
                print e.message