# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import stats

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


malf_lesions = {}
malf_thresh = {}
malf_tissues = {}
malf_lesion_locations = {}
malf_lesion_locations_nonzero = {}

for mod in modalities:
    malf_lesions[mod] = {}
    for m in malf_classes:
        malf_lesions[mod][m] = []
    
for img in mri_list:
    print img.priors['t1p']
    lesions = nib.load(img.lesions).get_data()

    tissues = {}
    for mod in modalities:
        tissues[mod] = nib.load(img.priors[mod]).get_data()    
    
    for m in malf_classes:
        malf_tissues[m] = nib.load(img.malf[m]).get_data()
        malf_thresh[m] = np.greater_equal(malf_tissues[m], 0.7)
        malf_lesion_locations[m] = np.multiply(lesions, malf_thresh[m])
        malf_lesion_locations_nonzero[m] = np.nonzero(malf_lesion_locations[m])
    
    for m in malf_classes:
        for mod in modalities:
            for lesion_voxel in tissues[mod][malf_lesion_locations_nonzero[m]]:
                malf_lesions[mod][m].append(lesion_voxel)
  
plt.close('all')
f, subplots = plt.subplots(len(modalities), len(good_malf_classes), sharex=True)

for i, mod in enumerate(modalities):
    for j, m in enumerate(good_malf_classes):
        try:
            print i,j, len(malf_lesions[mod][m])
            kde = stats.gaussian_kde(malf_lesions[mod][m])
            X_plot = np.linspace(0, 1500, 1000)
            density = kde(X_plot)
                        
            subplots[i,j].plot(X_plot, density)
            subplots[0,j].set_title(m)
            subplots[i,0].set_ylabel(mod, rotation=0, size='large')
            #subplots[i,j].set_title('lesions in ' + mod + ' for ' + m + ', ' + malf_lesions[mod][m] + ' lesions')
     
            #plt.title(m + ', ' + str(np.shape(malf_lesions[m])[0]) + ' voxels')
    
            #plt.savefig(data_dir + m +'.jpg')
        
        except Exception as e:
            print 'couldnt do {0}'.format(m)
            print e.message

#plt.xlabel('voxel intensity')
#plt.ylabel('proportion of voxels')
#plt.show()
f.tight_layout()
plt.savefig(data_dir + 'plots.jpg')
plt.show()
#plt.close()

print 'done'