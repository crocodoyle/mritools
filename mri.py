data_dir = 'C:/MRI/MS-LAQ/'

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
