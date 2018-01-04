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

lbpRadii = [1,2,3]
riftRadii = [1,2,3]
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
    outcomes = {}
    for metric in metrics:
        outcomes[metric] = []
        for scan in mri_list:
            if metric == 'newT1':
                outcomes[metric].append(scan.newT1)
            elif metric == 'newT2':
                outcomes[metric].append(scan.newT2)
            elif metric == 'newT1andT2':
                outcomes[metric].append(scan.newT1andT2)

    return outcomes


# removes extremely rare lesion-types
def prune_features(trainData, testData):
    featureCounts = {}
    for s, size in enumerate(sizes):
        featureCounts[size] = np.zeros((np.shape(trainData[size])[1]))

    for s, size in enumerate(sizes):
        testData[size] = testData[size][:, (trainData[size] != 0).sum(axis=0) >= 10]
        trainData[size] = trainData[size][:, (trainData[size] != 0).sum(axis=0) >= 10]

    return trainData, testData


def loadLesionNumbers(mri_list):
    featureVector = np.zeros((len(mri_list), 4))
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
            if len(lesion) > 100:
                featureVector[0] += 1
            elif len(lesion) > 25:
                featureVector[1] += 1
            elif len(lesion) > 10:
                featureVector[2] += 1
            elif len(lesion) > 2:
                featureVector[3] += 1
            
    return featureVector


def loadShape(mri_list, numLesions):
    data = defaultdict(list)
    
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                lesion_feature = pickle.load(open(scan.features_dir + 'shape_' + str(j) + '.pkl', 'rb'))

            if len(scan.lesionList[j]) > 100:
                data['large'].append(lesion_feature)
            elif len(scan.lesionList[j]) > 25:
                data['medium'].append(lesion_feature)
            elif len(scan.lesionList[j]) > 10:
                data['small'].append(lesion_feature)
            elif len(scan.lesionList[j]) > 2:
                data['tiny'].append(lesion_feature)
    
    if plotFeats:
        fig, ax = plt.subplots(1,4, figsize=(10,4))
        for s, size in enumerate(sizes):
            for d in data[size]:
                ax[s].plot(np.ndarray.flatten(d))
            
            ax[s].set_title('shape - ' + size)
        
        plt.tight_layout()
        plt.show()
            
    return data


def loadIntensity(mri_list, numLesions):
    numBins = 2
    
    data = defaultdict(list)

    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                lesion_feature = pickle.load(open(scan.features_dir + 'intensity_' + str(j) + '.pkl', 'rb'))
            else:
                continue
                
            if len(scan.lesionList[j]) > 100:
                feature = np.zeros((len(modalities), numBins))

                for m, mod in enumerate(modalities):
                    feature[m, :] = pickle.loads(lesion_feature[mod])
                data['large'].append(feature)
                
            elif len(scan.lesionList[j]) > 25:
                feature = np.zeros((len(modalities), numBins))

                for m, mod in enumerate(modalities):
                    feature[m, :] = pickle.loads(lesion_feature[mod])
                data['medium'].append(feature)
            
            elif len(scan.lesionList[j]) > 10:
                feature = np.zeros((len(modalities), numBins))

                for m, mod in enumerate(modalities):
                    feature[m, :] = pickle.loads(lesion_feature[mod])
                data['small'].append(feature)
            
            elif len(scan.lesionList[j]) > 2:
                feature = np.zeros((len(modalities), numBins))

                for m, mod in enumerate(modalities):
                    feature[m, :] = pickle.loads(lesion_feature[mod])
                data['tiny'].append(feature)

    if plotFeats:
        fig, ax = plt.subplots(1,4, figsize=(10,4))
        for s, size in enumerate(sizes):
            for d in data[size]:
                ax[s].plot(np.ndarray.flatten(d))
            
            ax[s].set_title('intensity - ' + size)
        
        plt.tight_layout()
        plt.show()
        
    return data
    
def loadRIFT(mri_list, numLesions):
    numBinsTheta = 8
    numBinsPhi = 1
       
#    data = np.zeros((numLesions, len(modalities), len(riftRadii), 8, numBinsTheta*numBinsPhi))
    data = defaultdict(list)
    
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                lesion_feature = pickle.load(open(scan.features_dir + 'rift_' + str(j) + '.pkl', 'rb'))
            else:
                continue
                
            if len(scan.lesionList[j]) > 100:
                feature = np.zeros((len(modalities), len(riftRadii), numBinsTheta*numBinsPhi))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(lesion_feature[mod])
                data['large'].append(feature)

            elif len(scan.lesionList[j]) > 25:
                feature = np.zeros((len(modalities), len(riftRadii), numBinsTheta*numBinsPhi))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(lesion_feature[mod])
                data['medium'].append(feature)

            elif len(scan.lesionList[j]) > 10:
                feature = np.zeros((len(modalities), len(riftRadii), numBinsTheta*numBinsPhi))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(lesion_feature[mod])
                data['small'].append(feature)
                
            elif len(scan.lesionList[j]) > 2:
                feature = np.zeros((len(modalities), len(riftRadii), numBinsTheta*numBinsPhi))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(lesion_feature[mod])
                data['tiny'].append(feature)

    if plotFeats:
        fig, ax = plt.subplots(1,4, figsize=(10,4))
        for s, size in enumerate(sizes):
            for d in data[size]:
                ax[s].plot(np.ndarray.flatten(d))
            
            ax[s].set_title('RIFT - ' + size)
        
        plt.tight_layout()
        plt.show()
        
    return data


def loadContext(mri_list, numLesions):
    numBins = 2
    
    data = defaultdict(list)
        
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
            if len(scan.lesionList[j]) > 2:
                lesion_feature = pickle.load(open(scan.features_dir + 'context_' + str(j) + '.pkl', 'rb'))
            else:
                continue
                    
            if len(scan.lesionList[j]) > 100:
                feature = np.zeros((len(tissues), numBins))

                for j, tissue in enumerate(scan.tissues):
                    feature[j, ...] = pickle.loads(lesion_feature[tissue])
                data['large'].append(feature)

            elif len(scan.lesionList[j]) > 25:
                feature = np.zeros((len(tissues), numBins))

                for j, tissue in enumerate(scan.tissues):
                    feature[j, ...] = pickle.loads(lesion_feature[tissue])
                data['medium'].append(feature)
            
            elif len(scan.lesionList[j]) > 10:
                feature = np.zeros((len(tissues), numBins))

                for j, tissue in enumerate(scan.tissues):
                    feature[j, ...] = pickle.loads(lesion_feature[tissue])
                data['small'].append(feature)
                
            elif len(scan.lesionList[j]) > 2:
                feature = np.zeros((len(tissues), numBins))

                for j, tissue in enumerate(scan.tissues):
                    feature[j, ...] = pickle.loads(lesion_feature[tissue])
                data['tiny'].append(feature)

    if plotFeats:
        fig, ax = plt.subplots(1,4, figsize=(10,4))
        for s, size in enumerate(sizes):
            for d in data[size]:
                ax[s].plot(np.ndarray.flatten(d))
            
            ax[s].set_title('context - ' + size)
        
        plt.tight_layout()
        plt.show()
            
    return data

def loadLBP(mri_list, numLesions, lbpPCA=None):
    #786 is 256*3
#    data = np.zeros((numLesions, len(modalities), len(lbpRadii), 8, 2**8), dtype='float')
    data = defaultdict(list)
            
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                lesion_feature = pickle.load(open(scan.features_dir + 'lbp_' + str(j) + '.pkl', 'rb'))
            else:
                continue

            if len(scan.lesionList[j]) > 100:
                feature = np.zeros((len(modalities), len(lbpRadii), 9))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(lesion_feature[mod])
                data['large'].append(feature)

            elif len(scan.lesionList[j]) > 25:
                feature = np.zeros((len(modalities), len(lbpRadii), 9))

                for m, mod in enumerate(modalities):
                    feature[m,...] = pickle.loads(lesion_feature[mod])
                data['medium'].append(feature)

            elif len(scan.lesionList[j]) > 10:
                feature = np.zeros((len(modalities), len(lbpRadii), 9))

                for m, mod in enumerate(modalities):
                    feature[m,...] = pickle.loads(lesion_feature[mod])
                data['small'].append(feature)
                
            elif len(scan.lesionList[j]) > 2:
                feature = np.zeros((len(modalities), len(lbpRadii), 9))

                for m, mod in enumerate(modalities):
                    feature[m,...] = pickle.loads(lesion_feature[mod])
                data['tiny'].append(feature)
    
    if plotFeats:
        fig, ax = plt.subplots(1,4, figsize=(14,4))
        for s, size in enumerate(sizes):
            for d in data[size]:
                ax[s].plot(np.ndarray.flatten(d))
            
            ax[s].set_title('LBP - ' + size)
        
        plt.tight_layout()
        plt.show()
            
    return data, lbpPCA


def loadAllData(mri_list, numLesions, lbpPCA=None):
    dataVectors = []
    print('loading context...')
    dataVectors.append(loadContext(mri_list, numLesions))
    print('loading rift...')
    dataVectors.append(loadRIFT(mri_list, numLesions))
    print('loading LBP...')
    dataVectors.append(loadLBP(mri_list, numLesions, lbpPCA=lbpPCA)[0])
#    dataVectors.append(loadGabor(mri_list, numLesions))
    print('loading shape...')
#    dataVectors.append(loadShape(mri_list, numLesions))
    print('loading intensity...')
    dataVectors.append(loadIntensity(mri_list, numLesions))

    for i in range(len(dataVectors)):
        for size in sizes:
            oneDataSourceDims = 1
            for dim in np.shape(dataVectors[i][size]):
                oneDataSourceDims *= dim
            oneDataSourceDims /= np.shape(dataVectors[i][size])[0]
            
            dataVectors[i][size] = np.reshape(np.vstack(dataVectors[i][size]), (np.shape(dataVectors[i][size])[0], oneDataSourceDims))

    return dataVectors, lbpPCA
    
    
def loadClinical(mri_list):
    new_mri_list, without_clinical = [], []
    for i, scan in enumerate(mri_list):
        clinicalData = pickle.load(open(scan.features_dir + 'clinical' + '.pkl', 'rb'))
        try:
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
#            print scan.uid, clinicalData
            without_clinical.append(scan)

#    writer = csv.writer(open('/usr/local/data/adoyle/bad_ids.csv', 'wb'))
#    for uid in bad_ids:
#        writer.writerow([uid[0:3] + '_' + uid[4:]])
        
    # print('we have', len(new_mri_list), 'patients with clinical data')
    # print('we have', len(without_clinical), 'patients without clinical data')

    return new_mri_list, without_clinical