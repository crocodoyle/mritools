import nibabel as nib
import numpy as np
import collections
import os

class mri(object):
    
    def __init__(self, t1p_image):
        
        tokens = t1p_image.split('_')
        self.data_dir = '/data1/users/adoyle/MS-LAQ/MS-LAQ-302-STX/'

        self.folder = self.data_dir + tokens[1] + '/' + tokens[2] + '_' + tokens[3] + '/m0/'
        self.features_dir = self.folder[:-3] + 'results'
        os.makedirs(self.features_dir, exist_ok=True)
        
        self.uid = tokens[2] + tokens[3]

        self.images = collections.OrderedDict()        
        self.images['t1p'] = self.folder + 'classifier_files/' + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t1p_norm_ANAT-brain_ISPC-stx152lsq6.mnc.gz'
        self.images['t2w'] = self.folder + 'classifier_files/' + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t2w_norm_ANAT-brain_ISPC-stx152lsq6.mnc.gz'
        self.images['pdw'] = self.folder + 'classifier_files/' + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_pdw_norm_ANAT-brain_ISPC-stx152lsq6.mnc.gz'
        self.images['flr'] = self.folder + 'classifier_files/' + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_flr_norm_ANAT-brain_ISPC-stx152lsq6.mnc.gz'          
        
        self.rawImages = collections.OrderedDict()
        self.rawImages['t1p'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t1p_ISPC-stx152lsq6.mnc.gz'
        self.rawImages['t2w'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t2w_ISPC-stx152lsq6.mnc.gz'
        self.rawImages['pdw'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_pdw_ISPC-stx152lsq6.mnc.gz'
        self.rawImages['flr'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_flr_ISPC-stx152lsq6.mnc.gz'
        
        self.lesions = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_ct2f_ISPC-stx152lsq6.mnc.gz'

        self.transformToICBM = self.folder[0:-3] + 'stx152lsq6/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_patient_stx152lsq6-to-stx152lsq6_nl.xfm'

        self.lesionList = []
        self.tissues = ['csf', 'wm', 'gm', 'pv', 'lesion']
        self.priors = collections.OrderedDict()
        
        self.lesionPriorXfm = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t1p-to-stx152lsq6.xfm'

        for tissue in self.tissues:
            self.priors[tissue] = self.folder[0:-3] + 'stx152lsq6/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_patient_avg_ANAT-' + tissue + '-cerebrum_ISPC-stx152lsq6.mnc.gz'

        self.newT1 = 0
        self.newT2 = 0
        self.newT1and2 = 0
        self.atrophy = 0.0
        self.treatment = ''
        
        self.futureLabels = self.folder[0:-3] + '/m24/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m24_ct2f_ISPC-stx152lsq6.mnc.gz'
        self.newLesions = 0
        
        self.newT2 = self.data_dir + tokens[1] + '/' + tokens[2] + '_' + tokens[3] + '/m24/' + 'classifier_files/' + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m24_t2w_norm_ANAT-brain_ISPC-stx152lsq6.mnc.gz'
        
    def calculateNewLesions(self):
        lesionImage = nib.load(self.futureLabels).get_data()
        lesionLocations = list(np.asarray(np.nonzero(lesionImage)).T)
        
        connectedLesion = np.zeros((len(lesionLocations)))
        lesionList = []
        
        for i, (x,y,z) in enumerate(lesionLocations):
            for lesion in lesionList:
                for point in lesion:
                    if np.abs(x - point[0]) <= 1 and np.abs(y - point[1]) <= 1 and np.abs(z-point[2]) <= 1:
                        lesion.append([x,y,z])
                        connectedLesion[i] = True
                    if connectedLesion[i]:
                        break
            if not connectedLesion[i]:
                newLesion = [[x,y,z]]
                lesionList.append(newLesion)
        
        self.newLesions = len(lesionList)
        
    def separateLesions(self):
        lesionImage = nib.load(self.lesions).get_data()
        lesionLocations = list(np.asarray(np.nonzero(lesionImage)).T)

        connectedLesion = np.zeros((len(lesionLocations)))

        lesionList = []
        
        for i, (x, y, z) in enumerate(lesionLocations):
            for lesion in lesionList:
                for point in lesion:
                    if np.abs(x - point[0]) <= 1 and np.abs(y - point[1]) <= 1 and np.abs(z - point[2]) <= 1:
                        lesion.append([x, y, z])
                        connectedLesion[i] = True
                    if connectedLesion[i]:
                        break
            
            if not connectedLesion[i]:
                newLesion = [[x,y,z]]
                lesionList.append(newLesion)
        
        self.lesionList = lesionList
        
        return lesionList