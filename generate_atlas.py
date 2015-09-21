from context_extraction import loadMRIList

import os, subprocess
import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt
from pyminc.volumes.factory import *

icbm = '/usr/local/data/adoyle/trials/quarantine/common/models/mni_icbm152_t1_tal_nlin_sym_09a.mnc'
icbm2 = '/usr/local/data/adoyle/trials/quarantine/common/models/icbm_avg_152_gm.mnc.gz'
output_dir = '/usr/local/data/adoyle/atlas/'

#mri_list = loadMRIList()
#for i, scan in enumerate(mri_list):
#    print i, '/', len(mri_list) 
#    outfile  = output_dir + scan.uid + '.mnc'
#    subprocess.call(['mincresample', '-transformation', scan.transformToICBM, '-like', icbm, '-tricubic', scan.lesions, outfile])
#
#

atlas = np.zeros((189,233,197), dtype='float')


for root, dirs, filenames in os.walk(output_dir):
    for f in filenames[0:10]:
        print root + f
        if f.endswith('.mnc'):
            image = nib.load(root + f).get_data()               
            atlas = atlas + np.divide(image, len(filenames), dtype='float')


atlas_image = nib.Nifti1Image(atlas, np.eye(4))
nib.save(atlas_image, '/usr/local/data/adoyle/trials/lesion_atlas.nii.gz')