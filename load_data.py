import numpy as np

import matplotlib.pyplot as plt
import pickle

from collections import defaultdict
import csv


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

wm_networks = ['Projection', 'Cerebellar', 'Optic', 'Cingulum', 'Inferior', 'Arcuate', 'Perisylvian',
               'Anterior_Commissure', 'Fornix', 'Corpus_Callosum']

lbpRadii = [1]


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
        new_lesions = int(scan.newT2)
        if new_lesions > 0:
            outcomes.append(1)
        else:
            outcomes.append(0)

    return outcomes


def loadIntensity(mri_list):
    numBins = 4
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

            feature = np.zeros((len(modalities), numBinsTheta))
            for m, mod in enumerate(modalities):
                feature[m, ...] = lesion_feature[mod]

            data.append(np.ndarray.flatten(feature))
        
    return np.asarray(data)


def loadContext(mri_list, include_catani):
    numBins = 2
    
    data = []
        
    for i, scan in enumerate(mri_list):
        context_priors = scan.tissues
        if include_catani:
            context_priors += wm_networks

        for j, lesion in enumerate(scan.lesionList):
            lesion_feature = pickle.load(open(scan.features_dir + 'context_' + str(j) + '.pkl', 'rb'))

            feature = np.zeros((len(context_priors), numBins), dtype='float32')

            for k, tissue in enumerate(context_priors):
                feature[k, :] = lesion_feature[tissue]
            data.append(np.ndarray.flatten(feature))

    data = np.asarray(np.asarray(data))
    print('data:', data[0])
    return data


def loadLBP(mri_list):
    data = []
            
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            lesion_feature = pickle.load(open(scan.features_dir + 'lbp_' + str(j) + '.pkl', 'rb'))

            feature = np.zeros((len(modalities), len(lbpRadii), 9))
            for m, mod in enumerate(modalities):
                feature[m, ...] = lesion_feature[mod]
            data.append(np.ndarray.flatten(feature))

    return np.asarray(np.asarray(data))


def loadAllData(mri_list, include_catani):

    context = loadContext(mri_list, include_catani)
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
            # print(scan.uid, clinicalData)

            scan.newT2 = int(clinicalData['newT2'])
            scan.newGad = int(clinicalData['gad'])

            if int(clinicalData['newT2']) > 0:
                scan.activity = 1
            else:
                scan.activity = 0

            scan.age = clinicalData['age']
            scan.country = clinicalData['country']
            scan.sex = clinicalData['sex']
            scan.race = clinicalData['race']

            scan.relapse = clinicalData['relapse']
            scan.treatment = clinicalData['treatment']

            new_mri_list.append(scan)
        except:
            without_clinical.append(scan)

    return new_mri_list, without_clinical


def load_responders(responder_filename, mri_list):
    found = 0
    with open(responder_filename) as responder_file:
        responder_mri_list = []
        responder_reader = csv.reader(responder_file)

        lines = list(responder_reader)

        for scan in mri_list:

            responder_info_found = False

            for line in lines:
                if line[0] in scan.uid:
                    print('Found responder info!')
                    responder = int(line[2])
                    scan.responder = responder
                    responder_info_found = True
                    found += 1

            if not responder_info_found:
                print('Couldnt find info for', scan.uid, 'on', scan.treatment)
                scan.responder = 0

            responder_mri_list.append(scan)

    print('Started with', len(mri_list), 'subjects, have responder info for', found)

    return responder_mri_list