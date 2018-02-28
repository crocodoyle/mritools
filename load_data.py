import numpy as np

import matplotlib.pyplot as plt
import pickle

from collections import defaultdict

modalities = ['t1p', 't2w', 'pdw', 'flr']
tissues = ['csf', 'wm', 'gm', 'pv', 'lesion']
metrics = ['newT2']
feats = ["Context", "RIFT", "LBP", "Intensity"]
sizes = ["tiny", "small", "medium", "large"]

scoringMetrics = ['TP', 'FP', 'TN', 'FN']

wm_tracts = ['Anterior_Segment', 'Arcuate', 'Cingulum', 'Cortico_Ponto_Cerebellum', 'Cortico_Spinal',
             'Inferior_Cerebellar_Pedunculus', 'Inferior_Longitudinal_Fasciculus',
             'Inferior_Occipito_Frontal_Fasciculus', 'Long_Segment', 'Optic_Radiations', 'Posterior_Segment',
             'Superior_Cerebelar_Pedunculus', 'Uncinate', 'Anterior_Commissure', 'Corpus_Callosum', 'Fornix', 'Internal_Capsule']


lbpRadii = [1]
riftRadii = [3, 6]
selectK = False
visualizeAGroup = False

letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)']

treatments = ['Placebo', 'Laquinimod', 'Avonex']

threads = 1

plotFeats = False
usePCA = False

def getLesionSizes(mri_list):
    numLesions = 0
    
    lesionSizes = []
    brainUids = []
    lesionCentroids = []
    
    print('Counting lesions')
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
            
            if len(lesion) > 2:            
                numLesions += 1
                lesionSizes.append(len(lesion))
                brainUids.append(scan.uid)
                
                x, y, z = [int(np.mean(x)) for x in zip(*lesion)]
                lesionCentroids.append((x, y, z))
    
    print('Counted lesions, we have', numLesions)
    return numLesions, lesionSizes, lesionCentroids, brainUids


def get_outcomes(mri_list):
    outcomes = []

    for scan in mri_list:
        outcomes.append(scan.newT2)

    return outcomes


def loadIntensity(mri_list):
    numBins = 2
    data = []

    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            lesion_feature = pickle.load(open(scan.features_dir + 'intensity_' + str(j) + '.pkl', 'rb'))

            feature = np.zeros((len(modalities), numBins))
            for m, mod in enumerate(modalities):
                feature[m, :] = lesion_feature[mod]

            data.append(np.ndarray.flatten(feature))
        
    return np.asarray(data)
    
def loadRIFT(mri_list):
    numBinsTheta = 4
    data = []
    
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            lesion_feature = pickle.load(open(scan.features_dir + 'rift_' + str(j) + '.pkl', 'rb'))

            feature = np.zeros((len(modalities), len(riftRadii), numBinsTheta))
            for m, mod in enumerate(modalities):
                feature[m, ...] = lesion_feature[mod]

            data.append(np.ndarray.flatten(feature))
        
    return np.asarray(data)


def loadContext(mri_list):
    numBins = 2
    
    data = []
        
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
            lesion_feature = pickle.load(open(scan.features_dir + 'context_' + str(j) + '.pkl', 'rb'))

            feature = np.zeros((len(tissues) + len(wm_tracts), numBins))

            for k, tissue in enumerate(scan.tissues + wm_tracts):
                feature[k, ...] = lesion_feature[tissue]
            data.append(np.ndarray.flatten(feature))
            
    return np.asarray(data)

def loadLBP(mri_list):
    #786 is 256*3
#    data = np.zeros((numLesions, len(modalities), len(lbpRadii), 8, 2**8), dtype='float')
    data = []
            
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            lesion_feature = pickle.load(open(scan.features_dir + 'lbp_' + str(j) + '.pkl', 'rb'))

            feature = np.zeros((len(modalities), len(lbpRadii), 9))
            for m, mod in enumerate(modalities):
                feature[m, ...] = lesion_feature[mod]
            data.append(np.ndarray.flatten(feature))

    return np.asarray(np.asarray(data))


def loadAllData(mri_list):

    context = loadContext(mri_list)
    rift = loadRIFT(mri_list)
    lbp = loadLBP(mri_list)
    intensity = loadIntensity(mri_list)

    size_feature = []
    for scan in mri_list:
        for lesion in scan.lesionList:
            size_feature.append(len(lesion))

    size_feature = np.reshape(size_feature, ((len(size_feature), 1))) / np.max(size_feature)

    print('Feature vector sizes:')
    print('Context:', context.shape)
    print('RIFT:', rift.shape)
    print('LBP:', lbp.shape)
    print('Intensity:', intensity.shape)
    print('Size:', size_feature.shape)

    feature_data = np.hstack((context, rift, lbp, intensity, size_feature))

    return feature_data
    
    
def loadClinical(mri_list):
    new_mri_list, without_clinical = [], []
    for i, scan in enumerate(mri_list):
        try:
            clinicalData = pickle.load(open(scan.features_dir + 'clinical.pkl', 'rb'))
            if int(clinicalData['newT1']) > 0:
                scan.newT1 = 1
            else:
                scan.newT1 = 0
                
            if int(clinicalData['newT2']) > 0:
                scan.newT2 = 1
            else:
                scan.newT2 = 0

            if scan.newT1 or scan.newT2:
                scan.newT1andT2 = 1
            else:
                scan.newT1andT2 = 0

            scan.atrophy = float(clinicalData['atrophy'])
            scan.treatment = clinicalData['treatment']
            new_mri_list.append(scan)
        except:
            without_clinical.append(scan)

    return new_mri_list, without_clinical