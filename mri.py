
import nibabel as nib
import numpy as np


#malf_classes = ['bg', 'bs', 'cgm', 'crblr_gm', 'crblr_wm', 'csf', 'dgm', 'lv', 'ov', 'wm']
#modalities = ['t1p', 't2w', 'pdw', 'flr']
#good_malf_classes = ['cgm', 'dgm', 'wm']


class mri(object):
    
    def __init__(self, t1p_image):
        
        tokens = t1p_image.split('_')
        self.data_dir = data_dir = '/usr/local/data/adoyle/trials/MS-LAQ-302-STX/'

        self.folder = data_dir + tokens[1] + '/' + tokens[2] + '_' + tokens[3] + '/m0/'
        
        self.uid = tokens[2] + tokens[3]

        self.images = {}        
        self.images['t1p'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t1p_ISPC-stx152lsq6.mnc.gz'
        self.images['t2w'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t2w_ISPC-stx152lsq6.mnc.gz'
        self.images['pdw'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_pdw_ISPC-stx152lsq6.mnc.gz'
        self.images['flr'] = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_flr_ISPC-stx152lsq6.mnc.gz'          
        
        self.lesions = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_ct2f_ISPC-stx152lsq6.mnc.gz'

        self.transformToICBM = self.folder[0:-3] + 'stx152lsq6/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_patient_stx152lsq6-to-stx152lsq6_nl.xfm'

        self.lesionList = []
        self.tissues = ['csf', 'wm', 'gm', 'lesion']
        self.priors = {}
        
        self.lesionPriorXfm = self.folder + 'MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t1p-to-stx152lsq6.xfm'

        for tissue in self.tissues:
            self.priors[tissue] = self.folder[0:-3] + 'stx152lsq6/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_patient_avg_ANAT-' + tissue + '-cerebrum_ISPC-stx152lsq6.mnc.gz'

    def separateLesions(self):
        lesionImage = nib.load(self.lesions).get_data()
        lesionLocations = list(np.asarray(np.nonzero(lesionImage)).T)
        
        lesionList = []

        for x, y, z in lesionLocations:
            connected = False
            for individualLesion in lesionList:
                for point in individualLesion:
                    if np.abs(x - point[0]) <= 1 and np.abs(y - point[1]) <= 1 and np.abs(z - point[2] <= 1) and not [x,y,z] in individualLesion:
                        individualLesion.append([x,y,z])
                        connected = True
            
            if not connected:
                newLesion = []
                newLesion.append([x,y,z])
                lesionList.append(newLesion)                        

        self.lesionList = lesionList

        return lesionList


#
#        while np.shape(lesionLocations)[0] > 0:      
#            oneLesion = []
#            location = lesionLocations.pop(0)            
#            x = location[0]
#            y = location[1]
#            z = location[2]
#            
#            oneLesion.append([x,y,z])
#             
#            i=0
#            keepGoing=True
#            while keepGoing:
#                spot = lesionLocations[i]
#                x = spot[0]
#                y = spot[1]
#                z = spot[2]
#                
#                print i, np.shape(lesionLocations)[0], x, y, z
#                
#                
#                for j, location in enumerate(oneLesion):
#                        if np.abs(x - location[0]) <= 1 and np.abs(y - location[1]) <= 1 and np.abs(z - location[2] <= 1) and not [x,y,z] in oneLesion:                                                
#                            oneLesion.append([x,y,z])
#                            lesionLocations.pop(i)
#                            i=0
#                            break
#                
#                i+=1
#                if i == np.shape(lesionLocations)[0]:
#                    keepGoing = False
#                    print 'got one lesion'
#                
#            lesionList.append(oneLesion)
#            
#        return lesionList
#        
        