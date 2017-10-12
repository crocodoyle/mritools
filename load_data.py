import numpy as np

import matplotlib.pyplot as plt
from pymongo import MongoClient
import cPickle as pickle


from collections import defaultdict


modalities = ['t1p', 't2w', 'pdw', 'flr']
tissues = ['csf', 'wm', 'gm', 'pv', 'lesion']
#metrics = ['newT1', 'newT2', 'newT1andT2'] 
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

dbIP = '132.206.73.115'
#dbIP = '127.0.0.1'
dbPort = 27017

plotFeats = False
usePCA = False

def getLesionSizes(mri_list):
    numLesions = 0
    
    lesionSizes = []
    brainUids = []
    lesionCentroids = []
    
    print 'Counting lesions'
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
            
            if len(lesion) > 2:            
                numLesions += 1
                lesionSizes.append(len(lesion))
                brainUids.append(scan.uid)
                
                x, y, z = [int(np.mean(x)) for x in zip(*lesion)]
                lesionCentroids.append((x, y, z))
                
    
    print 'Counted lesions, we have', numLesions

    return numLesions, lesionSizes, lesionCentroids, brainUids

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
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']

    data = defaultdict(list)
    
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                bsonData = db['shape'].find_one({'_id': scan.uid + '_' + str(j)})
                
            if len(scan.lesionList[j]) > 100:
                data['large'].append(pickle.loads(bsonData['shapeHistogram']))
            elif len(scan.lesionList[j]) > 25:
                data['medium'].append(pickle.loads(bsonData['shapeHistogram']))
            elif len(scan.lesionList[j]) > 10:
                data['small'].append(pickle.loads(bsonData['shapeHistogram']))
            elif len(scan.lesionList[j]) > 2:
                data['tiny'].append(pickle.loads(bsonData['shapeHistogram']))
    
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
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    numBins = 2
    
    data = defaultdict(list)
    
#    np.zeros((numLesions, len(modalities), numBins))
    
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                bsonData = db['intensity'].find_one({'_id': scan.uid + '_' + str(j)})
            else:
                continue
                
            if len(scan.lesionList[j]) > 100:
                feature = np.zeros((len(modalities), numBins))

                for m, mod in enumerate(modalities):
                    feature[m, :] = pickle.loads(bsonData[mod])
                data['large'].append(feature)
                
            elif len(scan.lesionList[j]) > 25:
                feature = np.zeros((len(modalities), numBins))

                for m, mod in enumerate(modalities):
                    feature[m, :] = pickle.loads(bsonData[mod])
                data['medium'].append(feature)
            
            elif len(scan.lesionList[j]) > 10:
                feature = np.zeros((len(modalities), numBins))

                for m, mod in enumerate(modalities):
                    feature[m, :] = pickle.loads(bsonData[mod])
                data['small'].append(feature)
            
            elif len(scan.lesionList[j]) > 2:
                feature = np.zeros((len(modalities), numBins))

                for m, mod in enumerate(modalities):
                    feature[m, :] = pickle.loads(bsonData[mod])
                data['tiny'].append(feature)

#    for m, mod in enumerate(modalities):
#        normalizingFactor = np.max(data[:, m])
#            
#        for f, feature in enumerate(data):
#            data[f, m] = np.divide(data[f, m], normalizingFactor)

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
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']

    numBinsTheta = 8
    numBinsPhi = 1
       
#    data = np.zeros((numLesions, len(modalities), len(riftRadii), 8, numBinsTheta*numBinsPhi))
    data = defaultdict(list)
    
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                bsonData = db['rift'].find_one({'_id': scan.uid + '_' + str(j)})
            else:
                continue
                
            if len(scan.lesionList[j]) > 100:
                feature = np.zeros((len(modalities), len(riftRadii), numBinsTheta*numBinsPhi))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(bsonData[mod])
                data['large'].append(feature)

            elif len(scan.lesionList[j]) > 25:
                feature = np.zeros((len(modalities), len(riftRadii), numBinsTheta*numBinsPhi))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(bsonData[mod])
                data['medium'].append(feature)

            elif len(scan.lesionList[j]) > 10:
                feature = np.zeros((len(modalities), len(riftRadii), numBinsTheta*numBinsPhi))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(bsonData[mod])
                data['small'].append(feature)
                
            elif len(scan.lesionList[j]) > 2:
                feature = np.zeros((len(modalities), len(riftRadii), numBinsTheta*numBinsPhi))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(bsonData[mod])
                data['tiny'].append(feature)

#    for m, mod in enumerate(modalities):
#        for r, rad in enumerate(riftRadii):
#            normalizingFactor = np.max(data[:, m, r, :])
#            
#            for f, feature in enumerate(data):
#                data[f, m, r, :] = np.divide(data[f, m, r, :], normalizingFactor)
    
    if plotFeats:
        fig, ax = plt.subplots(1,4, figsize=(10,4))
        for s, size in enumerate(sizes):
            for d in data[size]:
                ax[s].plot(np.ndarray.flatten(d))
            
            ax[s].set_title('RIFT - ' + size)
        
        plt.tight_layout()
        plt.show()
        
    return data

#def loadGabor(mri_list, numLesions):
#    dbClient = MongoClient(dbIP, dbPort)
#    db = dbClient['MSLAQ']
#    
#    data = np.zeros((numLesions, len(modalities), 4*2*3), dtype='float')
#    
#    lesionIndex = 0
#    for i, scan in enumerate(mri_list):
#        for l, lesion in enumerate(scan.lesionList):
#            if len(lesion) > 2:
#                bsonData = db['gabor'].find_one({'_id': scan.uid + '_' + str(l)})
#                
#                for m, mod in enumerate(modalities):
#                    data[lesionIndex, m, :] = np.reshape(pickle.loads(bsonData[mod]), (4*2*3))
#                                
#                lesionIndex += 1
#        
#    return data
    
def loadContext(mri_list, numLesions):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    numBins = 2
    
    data = defaultdict(list)
        
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
            if len(scan.lesionList[j]) > 2:
                bsonData = db['context'].find_one({'_id': scan.uid + '_' + str(j)})
            else:
                continue
                    
            if len(scan.lesionList[j]) > 100:
                feature = np.zeros((len(tissues), numBins))

                for j, tissue in enumerate(scan.tissues):
                    feature[j, ...] = pickle.loads(bsonData[tissue])
                data['large'].append(feature)

            elif len(scan.lesionList[j]) > 25:
                feature = np.zeros((len(tissues), numBins))

                for j, tissue in enumerate(scan.tissues):
                    feature[j, ...] = pickle.loads(bsonData[tissue])
                data['medium'].append(feature)
            
            elif len(scan.lesionList[j]) > 10:
                feature = np.zeros((len(tissues), numBins))

                for j, tissue in enumerate(scan.tissues):
                    feature[j, ...] = pickle.loads(bsonData[tissue])
                data['small'].append(feature)
                
            elif len(scan.lesionList[j]) > 2:
                feature = np.zeros((len(tissues), numBins))

                for j, tissue in enumerate(scan.tissues):
                    feature[j, ...] = pickle.loads(bsonData[tissue])
                data['tiny'].append(feature)
                
#    for tissIndex in range(np.shape(data)[1]):
#        normalizingFactor = np.max(data[:,tissIndex])
#        
#        for d, dat in enumerate(data):
#            data[d,tissIndex] = np.divide(data[d,tissIndex], normalizingFactor)

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
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    #786 is 256*3
#    data = np.zeros((numLesions, len(modalities), len(lbpRadii), 8, 2**8), dtype='float')
    data = defaultdict(list)
            
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                bsonData = db['lbp'].find_one({'_id': scan.uid + '_' + str(j)})
            else:
                continue

            if len(scan.lesionList[j]) > 100:
                feature = np.zeros((len(modalities), len(lbpRadii), 9))

                for m, mod in enumerate(modalities):
                    feature[m, ...] = pickle.loads(bsonData[mod])
                data['large'].append(feature)

            elif len(scan.lesionList[j]) > 25:
                feature = np.zeros((len(modalities), len(lbpRadii), 9))

                for m, mod in enumerate(modalities):
                    feature[m,...] = pickle.loads(bsonData[mod])
                data['medium'].append(feature)

            elif len(scan.lesionList[j]) > 10:
                feature = np.zeros((len(modalities), len(lbpRadii), 9))

                for m, mod in enumerate(modalities):
                    feature[m,...] = pickle.loads(bsonData[mod])
                data['small'].append(feature)
                
            elif len(scan.lesionList[j]) > 2:
                feature = np.zeros((len(modalities), len(lbpRadii), 9))

                for m, mod in enumerate(modalities):
                    feature[m,...] = pickle.loads(bsonData[mod])
                data['tiny'].append(feature)
    
#    print np.shape(data)
#    data = data[np.all(data == 0, axis=3)]
#    print np.shape(data)
    
#    possiblePatterns = []
#    for i in range(768):
#        if np.sum(data[:,:,:,i]) != 0:
#            possiblePatterns.append(i)
#    
#    
#    
#    data = np.reshape(data[:,:,:,possiblePatterns], (numLesions, len(modalities)*len(lbpRadii)*len(possiblePatterns)))    
#    
#    print 'running pca on LBP data...'
#    if lbpPCA == None:
#        lbpPCA = PCA(n_components=0.95, whiten=True, copy=False)
#        compressedData = lbpPCA.fit_transform(data)
#    else:
#        compressedData = lbpPCA.transform(data)
#        

#    print 'explained variance ratio (LBP):', np.sum(lbpPCA.explained_variance_ratio_)
    
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
    print 'loading context...'
    dataVectors.append(loadContext(mri_list, numLesions))
    print 'loading rift...'
    dataVectors.append(loadRIFT(mri_list, numLesions))
    print 'loading LBP...'
    dataVectors.append(loadLBP(mri_list, numLesions, lbpPCA=lbpPCA)[0])
#    dataVectors.append(loadGabor(mri_list, numLesions))
    print 'loading shape...'
#    dataVectors.append(loadShape(mri_list, numLesions))
    print 'loading intensity...'
    dataVectors.append(loadIntensity(mri_list, numLesions))

    for i in range(len(dataVectors)):
        for size in sizes:
            oneDataSourceDims = 1
            for dim in np.shape(dataVectors[i][size]):
                oneDataSourceDims *= dim
            oneDataSourceDims /= np.shape(dataVectors[i][size])[0]
            
            dataVectors[i][size] = np.reshape(np.vstack(dataVectors[i][size]), (np.shape(dataVectors[i][size])[0], oneDataSourceDims))
    
#    normalization of each vector to sum to 1  
#    for i in range(len(dataVectors)):
#        for size in sizes:
#            for j in range(np.shape(dataVectors[i][size])[0]):
#                dataVectors[i][size][j,:] /= np.sum(dataVectors[i][size][j,:])
#    
    return dataVectors, lbpPCA
    
    
def loadClinical(mri_list):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    new_mri_list = []
    without_clinical = []
    for i, scan in enumerate(mri_list):
        clinicalData = db['clinical'].find_one({'_id': scan.uid})
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
        
    print 'we have', len(new_mri_list), 'patients with clinical data'
    print 'we have', len(without_clinical), 'patients without clinical data'
    return new_mri_list, without_clinical