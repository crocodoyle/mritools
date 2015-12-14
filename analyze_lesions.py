import cPickle as pkl
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier


from scipy.spatial.distance import euclidean


import time

import matplotlib.pyplot as plt
from pymongo import MongoClient
import cPickle as pickle

import vtk
import nibabel as nib

import matplotlib.cm as cm
import transformations as t
import os
import collections

modalities = ['t1p', 't2w', 'pdw', 'flr']
tissues = ['csf', 'wm', 'gm', 'lesion']
lbpRadii = [1,2,3]
riftRadii = [2,4,6]

threads = 8

dbIP = 'localhost'
dbPort = 27017

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


def loadRIFT(mri_list, numLesions):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']

    numBinsTheta = 8
    numBinsPhi = 4
    
#   feature = np.zeros((len(lesion), len(radii), numBinsTheta*numBinsPhi))
    
    data = np.zeros((numLesions, len(modalities), len(riftRadii), numBinsTheta*numBinsPhi))
    
    lesionIndex = 0
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                bsonData = db['rift'].find_one({'_id': scan.uid + '_' + str(j)})
                    
                for m, mod in enumerate(modalities):
                    data[lesionIndex, m, :, :] = pickle.loads(bsonData[mod])
                
                lesionIndex += 1


    for m, mod in enumerate(modalities):
        for r, rad in enumerate(riftRadii):
            normalizingFactor = np.max(data[:, m, r, :])
            
            for f, feature in enumerate(data):
                data[f, m, r, :] = np.divide(data[f, m, r, :], normalizingFactor)

    return data

def loadGabor(mri_list, numLesions):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    data = np.zeros((numLesions, len(modalities), 4*2*3), dtype='float')
    
    lesionIndex = 0
    for i, scan in enumerate(mri_list):
        for l, lesion in enumerate(scan.lesionList):
            if len(lesion) > 2:
                bsonData = db['gabor'].find_one({'_id': scan.uid + '_' + str(l)})
                
                for m, mod in enumerate(modalities):
                    data[lesionIndex, m, :] = np.reshape(pickle.loads(bsonData[mod]), (4*2*3))
                                
                lesionIndex += 1
        
    return data
    
def loadContext(mri_list, numLesions):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    data = np.zeros((numLesions, 4))
    
    lesionIndex = 0
    for i, scan in enumerate(mri_list):
        for l, lesion in enumerate(scan.lesionList):
            if len(scan.lesionList[l]) > 2:
                bsonData = db['context'].find_one({'_id': scan.uid + '_' + str(l)})                
                j=0
                
                for tissue in scan.tissues:
                    if not 'pv' in tissue:
                        try:
                            data[lesionIndex, j] = pickle.loads(bsonData[tissue])
                            j+=1
                        except:
                            pass
                lesionIndex+=1 
    
    for tissIndex in range(np.shape(data)[1]):
        normalizingFactor = np.max(data[:,tissIndex])
        
        for d, dat in enumerate(data):
            data[d,tissIndex] = np.divide(data[d,tissIndex], normalizingFactor)
                
    return data

def loadLBP(mri_list, numLesions):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    lbpRadii = [1,2,3]

    data = np.zeros((numLesions, len(modalities), len(lbpRadii), 32), dtype='float')
    
    lesionIndex = 0
        
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            if len(scan.lesionList[j]) > 2:
                bsonData = db['lbp'].find_one({'_id': scan.uid + '_' + str(j)})
    
                for k, mod in enumerate(modalities):
                    for l, radius in enumerate(lbpRadii):
                        data[lesionIndex, k, l, :] = pickle.loads(bsonData[mod][str(radius)])
                lesionIndex += 1
    
    return data

def cluster(data, numClusters):
#    print 'Clustering...'
    
    startTime = time.time()
    kmeans = KMeans(n_clusters=numClusters, n_jobs=threads, copy_x=False)
    kmeans.fit_predict(data)
    endTime = time.time()
    
    elapsed = endTime - startTime
    print "Total time elapsed:", elapsed/60, 'minutes'

    return kmeans
    
def spectralCluster(data, numClusters):
    startTime = time.time()
    spectralCluster = SpectralClustering(n_clusters=numClusters, affinity='nearest_neighbors', n_neighbors=5)
    
    spectralCluster.fit_predict(data)
    endTime = time.time()
    
    elapsed = endTime - startTime
    print "Total time elapsed:", elapsed/60, 'minutes'

    return spectralCluster   
    

def countLesionTypes(mri_list, clusters, flattenedData, numLesions):
    lesionTypes = np.zeros((len(mri_list), len(clusters)))
    
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
           clusterDistances = np.sum(np.square( clusters - lesion ) )
           lesionTypes[i, np.argmin(clusterDistances)] += 1.0 / float(len(scan.lesionList))

    return lesionTypes
    
    
def ics(data, k, n):
    nCluster = np.shape(data)[0]
    variance = np.var(data)
    dims = np.shape(data)[1]
    
    logLikelihood = -(nCluster/2)*np.log(2*np.pi) - ((nCluster*dims)/2)*np.log(variance) - (nCluster - k)/2 + nCluster*np.log(nCluster) - nCluster*np.log(n)
    
    bic = logLikelihood - k*dims*np.log(n)
    aic = logLikelihood - 2*dims
    return bic, aic

def compareLesionClusterings(stopAt):
    print 'Loading patient data'
    mri_list = pkl.load(open('/usr/local/data/adoyle/mri_list.pkl', 'rb'))
    print 'Loaded patient data'

    numLesions, lesionSizes = getLesionSizes(mri_list)
    lbpData = loadLBP(mri_list, numLesions)
    gaborData = loadGabor(mri_list, numLesions)
    riftData = loadRIFT(mri_list, numLesions)
    
    featureTypes = ['lbp', 'gabor', 'rift']    
    
    lbpRadii = [1,2,3]    
    
    print 'Reshaping...'
    flattenedLBPData = np.reshape(lbpData, (np.shape(lbpData)[0], 32*len(modalities)*len(lbpRadii)))
    flattenedGaborData = np.reshape(gaborData, (np.shape(gaborData)[0], 4*2*3*len(modalities)))
    flattenedRIFTData = np.reshape(riftData, (np.shape(riftData)[0], 3*len(modalities)*8*4))


    numClusters = []
    
    bics = {}
    aics = {}
    
    for feature in featureTypes:
        bics[feature] = []
        aics[feature] = []

    for i in range(3,stopAt):
        print i
        lbpResults = cluster(flattenedLBPData, i)
        gaborResults = cluster(flattenedGaborData, i)
        riftResults = cluster(flattenedRIFTData, i)
        
        clusterBics = {}
        clusterAics = {}
        
        for feature in featureTypes:
            clusterBics[feature] = []
            clusterAics[feature] = []
        
        for j in range(i):
            lbpClusterData = flattenedLBPData[lbpResults.labels_==j]
            information = ics(lbpClusterData, i, np.shape(flattenedLBPData)[0])
            
            clusterBics['lbp'].append(information[0])
            clusterAics['lbp'].append(information[1])            
            
            gaborClusterData = flattenedGaborData[gaborResults.labels_==j]
            information = ics(gaborClusterData, i, np.shape(flattenedGaborData)[0])

            clusterBics['gabor'].append(information[0])
            clusterAics['gabor'].append(information[1])  
            
            riftClusterData = flattenedRIFTData[riftResults.labels_==j]
            information = ics(riftClusterData, i, np.shape(flattenedRIFTData)[0])

            clusterBics['rift'].append(information[0])
            clusterAics['rift'].append(information[1])  
              
        numClusters.append(i)
        for feature in featureTypes:
            bics[feature].append(np.sum(clusterBics[feature]))
            aics[feature].append(np.sum(clusterAics[feature]))
            
            plt.plot(numClusters, bics[feature], label=feature + ' BIC')
#            plt.plot(numClusters, aics[feature], label=feature + 'AIC')    
            
            print 'optimal clusters for', feature, ':', numClusters[np.argmax(bics[feature])]

        plt.xlabel('# lesion clusters')
        plt.ylabel('information')
        plt.title('Evaluating K in K-means')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
    
    return numClusters[np.argmax(bics['lbp'])], numClusters[np.argmax(bics['gabor'])], numClusters[np.argmax(bics['rift'])]

def compareBrainClusterings(Ks):
    print 'Loading patient data'
    mri_list = pkl.load(open('/usr/local/data/adoyle/mri_list.pkl', 'rb'))
    print 'Loaded patient data'

    numLesions, lesionSizes = getLesionSizes(mri_list)
    lbpData = loadLBP(mri_list, numLesions)
    gaborData = loadGabor(mri_list, numLesions)
    riftData = loadRIFT(mri_list, numLesions)
#    contextData = loadContext(mri_list, numLesions)    
    
    
    lbpRadii = [1,2,3]    
    
    print 'Reshaping...'
    flattenedLBPData = np.reshape(lbpData, (np.shape(lbpData)[0], 32*len(modalities)*len(lbpRadii)))
    flattenedGaborData = np.reshape(gaborData, (np.shape(gaborData)[0], 4*2*3*len(modalities)))
    flattenedRIFTData = np.reshape(riftData, (np.shape(riftData)[0], 3*len(modalities)*8*4))

    lbpResults = cluster(flattenedLBPData, Ks['lbp'])
    gaborResults = cluster(flattenedGaborData, Ks['gabor'])
    riftResults = cluster(flattenedRIFTData, Ks['rift'])

    
    
    textureType = np.zeros((len(mri_list), Ks['lbp'] + Ks['gabor'] + Ks['rift'] + 6))    
            
    lesionIndex = 0
    for i, brain in enumerate(mri_list):
        numLesions = len(brain.lesionList)
        
        for lbpType, gaborType, riftType in zip(lbpResults.labels_[lesionIndex:lesionIndex+numLesions], gaborResults.labels_[lesionIndex:lesionIndex+numLesions], riftResults.labels_[lesionIndex:lesionIndex+numLesions]):
            textureType[i, lbpType] += 1
            textureType[i, Ks['lbp'] + gaborType] += 1
            textureType[i, Ks['lbp'] + Ks['gabor'] + riftType] += 1
        
        for lesion in brain.lesionList:
            lesionSize = len(lesion)
            
            if lesionSize <= 10:
                textureType[i, Ks['lbp'] + Ks['gabor'] + Ks['rift'] + 1] += 1.0 / numLesions
            elif lesionSize <= 20:
                textureType[i, Ks['lbp'] + Ks['gabor'] + Ks['rift'] + 2] += 1.0 / numLesions
            elif lesionSize <= 50:
                textureType[i, Ks['lbp'] + Ks['gabor'] + Ks['rift'] + 3] += 1.0 / numLesions
            elif lesionSize <= 100:
                textureType[i, Ks['lbp'] + Ks['gabor'] + Ks['rift'] + 4] += 1.0 / numLesions
            else:
                textureType[i, Ks['lbp'] + Ks['gabor'] + Ks['rift'] + 5] += 1.0 / numLesions
                
        
        # take the frobenius norm
#        textureType[i, :] = np.linalg.norm(lesionTypeFeature[i,:])

        lesionIndex += numLesions    
    
    bics = []
    numClusters = []
    
    for i in range(1,24):
        print i
        results = cluster(textureType, i)
        
        clusterBics = []
        numClusters.append(i)
        
        for j in range(i):
            clusterData = textureType[results.labels_==j]
            information = ics(clusterData, i, np.shape(clusterData)[0])
            
            clusterBics.append(information[0])
            
        bics.append(np.sum(clusterBics))
        
        print 'optimal clusters:', numClusters[np.argmax(bics)]
        
        plt.plot(numClusters, bics)

        plt.xlabel('# brain clusters')
        plt.ylabel('information')
        plt.title('Evaluating K')
        plt.show()
    
def lesionType(Ks):
    mri_list = pkl.load(open('/usr/local/data/adoyle/mri_list.pkl', 'rb'))
    
    numLesions, lesionSizes = getLesionSizes(mri_list)
    lbpData = loadLBP(mri_list, numLesions)
    gaborData = loadGabor(mri_list, numLesions)
    riftData = loadRIFT(mri_list, numLesions)         

    lbpRadii = [1,2,3]    
    
#    print 'Reshaping...'
    flattenedLBPData = np.reshape(lbpData, (np.shape(lbpData)[0], 32*len(modalities)*len(lbpRadii)))
    flattenedGaborData = np.reshape(gaborData, (np.shape(gaborData)[0], 4*2*3*len(modalities)))
    flattenedRIFTData = np.reshape(riftData, (np.shape(riftData)[0], 3*len(modalities)*8*4))

    lbpResults = cluster(flattenedLBPData, Ks['lbp'])
    gaborResults = cluster(flattenedGaborData, Ks['gabor'])
    riftResults = cluster(flattenedRIFTData, Ks['rift'])
    
    textureType = np.zeros((numLesions, Ks['lbp'], Ks['gabor'], Ks['rift']), dtype=np.float32)

    
    lesionIndex = 0
    for i, brain in enumerate(mri_list):
        numLesions = len(brain.lesionList)
        
        j = 0
        for lbpType, gaborType, riftType in zip(lbpResults.labels_[lesionIndex:lesionIndex+numLesions], gaborResults.labels_[lesionIndex:lesionIndex+numLesions], riftResults.labels_[lesionIndex:lesionIndex+numLesions]):
            textureType[lesionIndex + j, lbpType, gaborType, riftType] += 1.0 / numLesions
            j += 1
            
        lesionIndex += numLesions

    del lbpResults
    del gaborResults
    del riftResults
    
    del mri_list

    flattenedTextureType = np.reshape(textureType, (np.shape(textureType)[0], np.shape(textureType)[1]*np.shape(textureType)[2]*np.shape(textureType)[3]))
    del textureType

    bics = []
    numClusters = []
    
    for i in range(3,60):
        print i
        results = cluster(flattenedTextureType, i)
        
        clusterBics = []
        numClusters.append(i)
        
        for j in range(i):
            clusterData = flattenedTextureType[results.labels_==j]
            information = ics(clusterData, i, np.shape(clusterData)[0])
            
            clusterBics.append(information[0])
            
        bics.append(np.sum(clusterBics))
        
        print 'optimal clusters:', numClusters[np.argmax(bics)]
        
        plt.plot(numClusters, bics)

        plt.xlabel('# brain clusters')
        plt.ylabel('information')
        plt.title('Evaluating K')
        plt.show()


def normalizeDataVectors(dataVectors):    
    dimensions = 0
    
    for i in range(len(dataVectors)):
        
        oneDataSourceDims = 1
        for dim in np.shape(dataVectors[i]):
            oneDataSourceDims *= dim
        oneDataSourceDims /= np.shape(dataVectors[i])[0]
        
        dimensions += oneDataSourceDims
        dataVectors[i] = np.reshape(dataVectors[i], (np.shape(dataVectors[i])[0], oneDataSourceDims))

        print 'feature dimensions:', dimensions
        
        
        
    for i in range(len(dataVectors)):
        otherDims = dimensions - np.shape(dataVectors[i])[1]
        
        if not otherDims == 0:
            dataVectors[i] = np.multiply(dataVectors[i], float(otherDims)/float(dimensions))
    
    if len(dataVectors) == 1:
        data = dataVectors[0]
    if len(dataVectors) == 2:
        data = np.hstack((dataVectors[0], dataVectors[1]))
    if len(dataVectors) == 3:
        data = np.hstack((dataVectors[0], dataVectors[1], dataVectors[2]))
    if len(dataVectors) == 4:
        data = np.hstack((dataVectors[0], dataVectors[1], dataVectors[2], dataVectors[3]))
    
    
    return data

def getNClosest(candidate, n, allLesionFeatures):

    distance = np.zeros((np.shape(allLesionFeatures)[0]))

    for i, lesionFeatures in enumerate(allLesionFeatures):
        distance[i] = euclidean(candidate, lesionFeatures)
    
    nClosest = distance.argsort()[:n+1]

    return nClosest
    
    
def hierarchicalClustering():
    print 'Loading patient data'
    mri_list = pkl.load(open('/usr/local/data/adoyle/mri_list.pkl', 'rb'))
    print 'Loaded patient data'
        
    numLesions, lesionSizes, lesionCentroids, brainUids = getLesionSizes(mri_list)

    experiments = ['context', 'rift', 'lbp', 'context-lbp', 'context-rift', 'rift-lbp', 'context-lbp-rift']
      
    masks = []
    tinyMask = np.asarray(lesionSizes) <= 10
    smallMask = (np.asarray(lesionSizes) > 10) & (np.asarray(lesionSizes) <= 25)
    medMask = (np.asarray(lesionSizes) > 25) & (np.asarray(lesionSizes) <= 100)
    bigMask = np.asarray(lesionSizes) > 100
        
    masks.append(tinyMask)
    masks.append(smallMask)
    masks.append(medMask)
    masks.append(bigMask)
    
    maskName = ['tiny', 'small', 'medium', 'large']    
    
    
    for experiment in experiments:  
        contextData = loadContext(mri_list, numLesions)
        lbpData = loadLBP(mri_list, numLesions)
    #    gaborData = loadGabor(mri_list, numLesions)
        riftData = loadRIFT(mri_list, numLesions)        
            
        dataVectors = []
        labels = []
        if 'context' in experiment:
            dataVectors.append(contextData)
            labels.append(('context', 4))
        if 'rift' in experiment:
            dataVectors.append(riftData)
            labels.append(('rift', ))
        if 'lbp' in experiment:
            dataVectors.append(lbpData)
            labels.append('lbp')
            
        data = normalizeDataVectors(dataVectors)
        
        del contextData
        del dataVectors
        del lbpData
    #    del gaborData
        del riftData

        queryLesionIndices = []      
        
        for m, mask in enumerate(masks):
            lesionHighLevel = data[mask]
            
            uids = np.asarray(brainUids)[mask]
            centroids = np.asarray(lesionCentroids)[mask]
            sizes = np.asarray(lesionSizes)[mask]
            
            #  to choose k
    #        for i in range(1, 20):
    #            results = cluster(lesionHighLevel, i)
    #    #        results = spectralCluster(contextData, i)
    #            clusterBics = []
    #            numClusters.append(i)
    #            
    #            for j in range(i):
    #                clusterData = lesionHighLevel[results.labels_==j]
    #                information = ics(clusterData, i, np.shape(clusterData)[0])
    #                
    #                clusterBics.append(information[0])
    #                
    #            bics.append(np.sum(clusterBics))
    #            
    #            print 'optimal clusters:', numClusters[np.argmax(bics)]
    #            print numClusters
    #            
    #        plt.plot(numClusters, bics)
    #
    #        plt.xlabel('# brain clusters')
    #        plt.ylabel('information')
    #        plt.title('Evaluating K')
    #        plt.show()
    #    
    #        optimalClusters = numClusters[np.argmax(bics)]
    #        finalResults.append(cluster(contextData, optimalClusters))
    #        labelNum += optimalClusters
            
            
            optimalClusters = [3,3,3,3]
            results = cluster(lesionHighLevel, optimalClusters[m])        
            distances = results.transform(lesionHighLevel)

            featureSelection(lesionHighLevel, results, experiment, maskName[m])
            
            exampleLesions = []
            clusterDistance = []
            features = []
            
            for clusters in range(optimalClusters[m]):
                exampleLesions.append([])
                clusterDistance.append([])
                features.append([])
                queryLesionIndices.append([])
            
            for i, lesionCluster in enumerate(results.labels_):
                for lesionType in range(optimalClusters[m]):
                    if lesionCluster == lesionType:
                        exampleLesions[lesionCluster].append((uids[i], centroids[i], sizes[i]))
                        clusterDistance[lesionCluster].append(distances[i, lesionCluster])
                        features[lesionCluster].append(lesionHighLevel[i])
    
    
            for lesionCluster in range(optimalClusters[m]):
                examples = 1
                
                n = 5
#                nBest = np.asarray(clusterDistance[lesionCluster]).argsort()[:n]
                
                if os.path.exists('/usr/local/data/adoyle/lesionIndices.pkl'):
                    queryLesionIndices = pkl.load(open('/usr/local/data/adoyle/lesionIndices.pkl'))
                else:
                    query = np.random.randint(len(features[lesionCluster]))
                    for i, feature in enumerate(lesionHighLevel):
                        if (feature == features[lesionCluster][query]).all():
                            queryLesionIndices[m].append(i)
    
    
#                print 'query: ', queryLesionIndices[m][lesionCluster]
#                print 'feature: ', lesionHighLevel[queryLesionIndices[m][lesionCluster]]
                
                nClosest = getNClosest(lesionHighLevel[queryLesionIndices[m][lesionCluster]], n, lesionHighLevel)
                n=len(nClosest)
                
                plt.figure()
                
                for goodOne in nClosest:
                    
                    for scan in mri_list:
                        if scan.uid in uids[goodOne]:
                            img = nib.load(scan.images['flr']).get_data()
                            t2img = nib.load(scan.images['t2w']).get_data()
                            
                            x, y, z = centroids[goodOne]
    
                            lesionMaskImg = np.zeros((np.shape(img)))
                            
                            for lesion in scan.lesionList:
                                xx, yy, zz = [int(np.mean(xxx)) for xxx in zip(*lesion)]
                                if [xx, yy, zz] == [x,y,z]:
                                    for point in lesion:
                                        lesionMaskImg[point[0], point[1], point[2]] = 1
                                    break
                            
                            maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)                      
                            maskSquare = np.zeros((np.shape(img)))
                            maskSquare[x, y-10:y+10, z+10] = 1
                            maskSquare[x, y-10:y+10, z-10] = 1
                            maskSquare[x, y-10, z-10:z+10] = 1
                            maskSquare[x, y+10, z-10:z+10] = 1
                            
                            square = np.ma.masked_where(maskSquare == 0, np.ones(np.shape(maskSquare))*5000)
    
                            lesionMaskPatch = maskImg[x, y-20:y+20, z-20:z+20]
                            ax = plt.subplot(4, n, examples)
                            ax.axis('off')
                            ax.imshow(img[x,20:200,20:200], cmap = plt.cm.gray, interpolation = 'nearest',origin='lower')
                            ax.imshow(maskImg[x, 20:200,20:200], cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.2, origin='lower')
                            ax.imshow(square[x, 20:200, 20:200], cmap = plt.cm.autumn, interpolation = 'nearest', origin='lower')
                            
                            ax2 = plt.subplot(4, n, examples+n)
                            ax2.axis('off')                        
                            ax2.imshow(t2img[x, 20:200, 20:200], cmap = plt.cm.gray, interpolation = 'nearest', origin='lower')
                            ax2.imshow(maskImg[x, 20:200,20:200], cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.2, origin='lower')
                            ax2.imshow(square[x, 20:200, 20:200], cmap = plt.cm.autumn, interpolation = 'nearest', origin='lower')
                            
                            
                            ax3 = plt.subplot(4, n, examples+2*n)
                            ax3.axis('off')
                            ax3.imshow(img[x, y-20:y+20, z-20:z+20], cmap = plt.cm.gray, interpolation = 'nearest', origin='lower')
                            ax3.imshow(lesionMaskPatch, cmap = plt.cm.autumn, alpha = 0.2, interpolation = 'nearest', origin='lower')
                            
                            ax4 = plt.subplot(4, n, examples + 3*n)
                            ax4.imshow(t2img[x, y-20:y+20, z-20:z+20], cmap = plt.cm.gray, interpolation = 'nearest', origin='lower')
                            ax4.imshow(lesionMaskPatch, cmap = plt.cm.autumn, alpha = 0.2, interpolation = 'nearest', origin='lower')
                            ax4.set_frame_on(False)
                            ax4.axes.get_yaxis().set_visible(False)
                            ax4.set_xticks([])
                            plt.xlabel(results.labels_[goodOne] + 1)
                            
                            examples +=1
                            break
                        
    #            print maskName[m], ' lesions, cluster:', lesionCluster
                
                title = 'closest lesions in ' + maskName[m][0] + str(lesionCluster + 1)
                plt.suptitle(title, fontsize=14)
                plt.subplots_adjust(wspace=0.01,hspace=0.01)
                plt.savefig('/usr/local/data/adoyle/images/closest-' + experiment + '-' + maskName[m] + str(lesionCluster) + '.png', dpi=600)
                plt.show()
                
        pkl.dump(queryLesionIndices, open('/usr/local/data/adoyle/lesionIndices.pkl', 'wb'))
    
#            plt.figure()
#            
#            n = n-1
##            print nBest
#            
#            examples = 1
#            for goodOne in nBest:
#                for scan in mri_list:
#                    if scan.uid in exampleLesions[lesionCluster][goodOne][0]:
#                        img = nib.load(scan.images['flr']).get_data()
#                        x, y, z = exampleLesions[lesionCluster][goodOne][1]
#
#                        lesionMaskImg = np.zeros((np.shape(img)))
#                        
#                        for lesion in scan.lesionList:
#                            xx, yy, zz = [int(np.mean(xxx)) for xxx in zip(*lesion)]
#                            if [xx, yy, zz] == [x,y,z]:
#                                for point in lesion:
#                                    lesionMaskImg[point[0], point[1], point[2]] = 1
#                                
##                                print 'lesion size:', len(lesion)
#                                break
##                        print 'lesions matched:', lesionsMatched
#                        
#                        maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)                      
#                        maskSquare = np.zeros((np.shape(img)))
#                        maskSquare[:, y-10:y+10, z+10] = 1
#                        maskSquare[:, y-10:y+10, z-10] = 1
#                        maskSquare[:, y-10, z-10:z+10] = 1
#                        maskSquare[:, y+10, z-10:z+10] = 1
#                        
#                        square = np.ma.masked_where(maskSquare == 0, np.ones(np.shape(maskSquare))*5000)
#                        
#                        lesionMaskPatch = maskImg[x, y-20:y+20, z-20:z+20]
#
#                        plt.subplot(2, n, examples)
#                        plt.axis('off')
#                        plt.imshow(img[x,20:180,20:180], cmap = plt.cm.gray)
#                        plt.subplots_adjust(wspace=0.01,hspace=0.01)
##                        plt.tight_layout()And Dodig wasn't just talking about innovators focused on the high-tech sector. He pointed out that technological breakthroughs also key to the success of traditional industries like manufacturing and natural resources.
#
#
#                        plt.imshow(maskImg[x, 20:180,20:180], cmap = plt.cm.autumn, alpha = 0.2)
#                        plt.imshow(square[x, 20:180, 20:180], cmap = plt.cm.autumn, interpolation = 'nearest')
#                        
#                        plt.subplot(2, n, examples+n)
#                        plt.axis('off')
#                        plt.imshow(img[x, y-20:y+20, z-20:z+20], cmap = plt.cm.gray)
#                        plt.imshow(lesionMaskPatch, cmap = plt.cm.autumn, alpha = 0.2)
#                        plt.subplots_adjust(wspace=0.01, hspace=0.01)
#                        
#                        examples +=1
#                        break
#                    
##            print maskName[m], ' lesions, cluster:', lesionCluster
#            
#            title = maskName[m] + ' lesions, cluster ' + str(lesionCluster + 1)
#            plt.suptitle(title, fontsize=14)
#            plt.savefig('/usr/local/data/adoyle/images/' + experiment + '-' + maskName[m] + str(lesionCluster) + '.png')
#            plt.show()
            
            

#    vizLabels = np.zeros((np.shape(contextData)[0]), dtype='int')
#    
#    labelNum = 0
#    labelIndex = 0
#    for mask, result in zip(masks, finalResults):
#        labels = result.labels_[mask]
#        vizLabels[labelIndex:labelIndex + len(labels)] = labels + int(labelNum)
#        labelIndex += len(labels)
#        labelNum += len(set(labels))
#        
#    vtkViz(contextData, vizLabels)
        
#        
#        lesionCategory = np.zeros((optimalClusters, 4))
#        
#        for j in range(optimalClusters):
#            lesionSizesInCluster = []
#            inClusterMask = np.reshape([results.labels_==j], numLesions)
#                  
#            for i, lesionSize in enumerate(lesionSizes):
#                if inClusterMask[i]:
#                    lesionSizesInCluster.append(lesionSize)
#    
#            print len(lesionSizesInCluster)
#            
#            for lesionSize in lesionSizesInCluster:
#                if lesionSize <= 20:
#                    lesionCategory[j,0] += 1
#                elif lesionSize <= 200:
#                    lesionCategory[j,1] += 1
#                elif lesionSize <= 1000:
#                    lesionCategory[j,2] += 1
#                else:
#                    lesionCategory[j,3] += 1
#    




#    ylabels = ['wm/gm 65/55', 'WM 95+', 'gm 85', 'wm/gm 50/50', 'wm/gm 85/35', 'nothing', 'csf/wm/gm 20/45/50', 'wm/gm 75/45', 'csf/wm/gm 15/70/35']
#    ylabels = ['wm/gm', 'wm/gm', 'wm/gm', 'wm/gm', 'wm/gm', 'wm/gm', 'wm/gm', 'wm/gm', 'wm/gm']
#
#    for i, c in enumerate(results.cluster_centers_):
#        if c[1] > 0.9:
#            ylabels[i] = '.96 white matter'
#        if c[0] > 0.2:
#            ylabels[i] = '.21 cerebral spinal fluid'
#        if c[2] > 0.8:
#            ylabels[i] = '.83 gray matter'
#        if c[0] > .1 and c[1] > 0.7 and c[2] > 0.3:
#            ylabels[i] = 'high prob for all tissues'
#        if c[1] > 0.8 and c[2] > 0.3:
#            ylabels[i] = '.87 white matter, .34 gray matter' 
#        if c[0] < 0.1 and c[1] < 0.1 and c[2] < 0.1:
#            ylabels[i] = 'all tissues < .02'
#    
#    xlabels = ['<20', '<200', '<1000', '1500+']    
#    
#    plt.imshow(lesionCategory, interpolation="nearest")
#    plt.colorbar()
#    plt.xlabel('lesion size')
#    plt.ylabel('lesion context')
#    plt.xticks(range(len(xlabels)), xlabels)
##    plt.yticks(range(9), ylabels)
#    plt.show()
    
def featureSelection(features, results, experiment, lesionSize):
    rf = RandomForestClassifier(n_estimators=50)
    
    rf.fit(features, results.labels_)
    
    plotIndex = 0
    plotWidth = 0    
    data = collections.OrderedDict()
    
    if 'context' in experiment:
        plotWidth += 2
        data['context'] = collections.OrderedDict()
        for i, tiss in enumerate(tissues):
            data['context'][tiss[0]] = collections.OrderedDict()
            data['context'][tiss[0]][tiss] = rf.feature_importances_[plotIndex]
            plotIndex += 1
            
    if 'lbp' in experiment:
        plotWidth += 80
        for m, mod in enumerate(modalities):
            data['lbp-' + mod] = collections.OrderedDict()
            for r, rad in enumerate(lbpRadii):
                data['lbp-' + mod][str(rad)] = collections.OrderedDict()
                for x in range(32):
                    data['lbp-'+mod][str(rad)][str(x)] = rf.feature_importances_[plotIndex]
                    plotIndex += 1
    
    if 'rift' in experiment:
        plotWidth += 80
        for m, mod in enumerate(modalities):
            data['rift-' + mod] = collections.OrderedDict()
            for r, rad in enumerate(riftRadii):
                data['rift-' + mod][str(rad)] = collections.OrderedDict()
                for x in range(32):
                    data['rift-' + mod][str(rad)][str(x)] = rf.feature_importances_[plotIndex]
                    plotIndex +=1
    
    
    fig = plt.figure(figsize=(plotWidth, 3), dpi=100)
    ax = fig.add_subplot(1,1,1)
    
    label_group_bar(ax, data)     
    
    fig.savefig('/usr/local/data/adoyle/images/featureImportance-' + experiment + '-' + lesionSize + '.png', dpi=100, bbox_inches='tight')
    fig.show()     

def mk_groups(data):
    try:
        newdata = data.items()
    except:
        return

    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = mk_groups(value)
        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))
            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups
    return [thisgroup] + groups

def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)

def label_group_bar(ax, data):
    groups = mk_groups(data)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = range(1, ly + 1)

    ax.bar(xticks, y, align='center')
    ax.set_xticks(xticks)
    ax.set_xticklabels(x)
    
    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        tick.label.set_rotation('vertical')

    scale = 1. / ly
    for pos in xrange(ly + 1):
        add_line(ax, pos * scale, -.1)
    ypos = -.2
    while groups:
        group = groups.pop()
        pos = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos * scale, ypos)
            pos += rpos
        add_line(ax, pos * scale, ypos)
        ypos -= .1


def vtkViz(features, clusterMembership):
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()    
    
    colours = vtk.vtkUnsignedCharArray()
    colours.SetNumberOfComponents(3)
    colours.SetName("Colours")    
    
    clusterColours = []
    for j in range(len(list(set(clusterMembership)))):
        clusterColours.append(np.random.randint(0, 255, 3))
    
    
    for i, point in enumerate(features):
#        print point
        pointId = points.InsertNextPoint(point)
        colour = clusterColours[clusterMembership[i]]
        colours.InsertNextTuple3(colour[0], colour[1], colour[2])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pointId)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(vertices)
    poly.GetCellData().SetScalars(colours)
    poly.Modified()
    poly.Update()


    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
     
    renWin.SetSize(500, 500)



#    transform = vtk.vtkTransform()
#    transform.Translate(1.0, 0.0, 0.0)
    
    axes = vtk.vtkAxesActor()
#    axes.SetUserTransform(transform)
    axes.SetXAxisLabelText("wm")
    axes.SetYAxisLabelText("gm")
    axes.SetZAxisLabelText("les")
#    axes.SetTotalLength(100, 100, 100)
#    axes.SetShaftTypeToCylinder()
#    axes.SetCylinderRadius(0.005)


    #delaunay = vtk.vtkDelaunay2D()
    #delaunay.SetInput(poly)

#    glyphFilter = vtk.vtkVertexGlyphFilter()
#    glyphFilter.SetInput(poly)
#    glyphFilter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(poly)
    
    actor = vtk.vtkActor()
    actor.GetProperty().SetPointSize(2)
    actor.SetMapper(mapper)
    
    ren.AddActor(actor)
    ren.AddActor(axes)
    ren.SetBackground(.2, .3, .4)
    
    renWin.Render()
    iren.Start()

if __name__ == "__main__":
    Ks = {}
    Ks['lbp'] = 12
    Ks['rift'] = 13
    Ks['gabor'] = 28
    
    
    hierarchicalClustering()
#    compareBrainClusterings(Ks)
#    lesionType(Ks)
#    compareLesionClusterings(120)