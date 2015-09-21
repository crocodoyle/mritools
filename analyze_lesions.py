import redis
import cPickle as pkl
import numpy as np

from mri import mri
from sklearn.cluster import KMeans

import simplejson as json
import time

import matplotlib.pyplot as plt

import sys


modalities = ['t1p', 't2w', 'pdw', 'flr']
scales = [1,2,3]


def getLesionSizes(mri_list):
    numLesions = 0
    
    lesionSize = []
    
    
    print 'Counting lesions'
    for i, scan in enumerate(mri_list):
    #    lesionSize.append(len(scan.lesionList))
        for lesion in enumerate(scan.lesionList):
            numLesions += 1
    
            for l in lesion:
                if not isinstance( l, ( int, long ) ):
                    lesionSize.append(len(l))
    
    print 'Counted lesions, we have', numLesions

    return numLesions, lesionSize


def loadLesionTextures(mri_list, numLesions):
    red = redis.StrictRedis(host='localhost', port=6379, db=0)

    data = np.zeros((numLesions, len(modalities), len(scales), 32), dtype='float')
    
    lesionIndex = 0
    
    jsonDecoder = json.decoder.JSONDecoder()
    
    print 'Loading lesion textures...'
    for i, scan in enumerate(mri_list):
        for j in range(len(scan.lesionList)):
            for k, mod in enumerate(modalities):
                for l, scale in enumerate(scales):
                    featureString = red.get(scan.uid + ':lbp:' + mod + ':scale' + str(scale) + ':lesion' + str(j))
                    
                    dataDecoded = jsonDecoder.decode(featureString)
                    
                    
                    data[lesionIndex, k, l, :] = dataDecoded
            lesionIndex += 1
    print 'Loaded textures'
    
    return data

def cluster(data, numClusters):
    print 'Clustering...'
    
    startTime = time.time()
    kmeans = KMeans(n_clusters=numClusters, n_jobs=8)
    kmeans.fit_predict(data)
    endTime = time.time()
    
    elapsed = endTime - startTime
    print "Total time elapsed:", elapsed/3600, 'hours', elapsed/60, 'minutes'

    return kmeans




#lesionSizeGroups = []
#
#for i in range(8):
#    print np.shape(flattenedData[kmeans.labels_==i])
#    lesionSizeGroups.append(np.asarray(lesionSize)[kmeans.labels_==i])
#    print 'std', np.std(lesionSizeGroups[i]), 'mean', np.mean(lesionSizeGroups[i])
#


def countLesionTypes(mri_list, clusters, flattenedData, numLesions):
    lesionTypes = np.zeros((len(mri_list), len(clusters)))
    
    lesionIndex = 0
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
           clusterDistances = np.sum(np.square( clusters - lesion ) )
           lesionTypes[i, np.argmin(clusterDistances)] += 1.0 / float(len(scan.lesionList))

    return lesionTypes


def compareLesionClusterings(stopAt):
    print 'Loading patient data'
    mri_list = pkl.load(open('/usr/local/data/adoyle/mri_list.pkl', 'rb'))
    print 'Loaded patient data'

    numLesions, lesionSizes = getLesionSizes(mri_list)
    data = loadLesionTextures(mri_list, numLesions)
    
    print 'Reshaping...'
    flattenedData = np.reshape(data, (np.shape(data)[0], 32*len(modalities)*len(scales)))

    allClustersError = []
    numClusters = []
    for i in range(3,stopAt):
        print i
        results = cluster(flattenedData, i)

        for j in range(i):
            localClusterError = np.zeros(i)

            clusterSize = np.shape(flattenedData[results.labels_==j])[0]
            
            localClusterError[j] = np.sum(((flattenedData[results.labels_==j] - results.cluster_centers_[j,:])**2) / clusterSize)       
        
        numClusters.append(i)            
#            for point in :
#                localClusterError[j] += np.sum(np.square()) / clusterSize
#            
        allClustersError.append(np.mean(localClusterError))
        plt.plot(numClusters, allClustersError)
        plt.show()    


def main():
    mri_list = pkl.load(open('/usr/local/data/adoyle/mri_list.pkl', 'rb'))
    
    numLesions, lesionSizes = getLesionSizes(mri_list)
    data = loadLesionTextures(mri_list, numLesions)
    
    flattenedData = np.reshape(data, (np.shape(data)[0], 32*len(modalities)*len(scales)))


    numLesionTypes = 100
    numBrains = len(mri_list)
    numBrainTypes = 5
    
    results = cluster(flattenedData, numLesionTypes)

    lesionTypeFeature = np.zeros((numBrains, numLesionTypes))

    print 'Counting lesion types...'
    lesionIndex = 0
    for i, brain in enumerate(mri_list):
#        print 'Patient', i
        numLesions = len(brain.lesionList)
        
        for lesionType in results.labels_[lesionIndex:lesionIndex+numLesions]:
            lesionTypeFeature[i, lesionType] += 1
        
        # take the frobenius norm
        lesionTypeFeature[i, :] = np.linalg.norm(lesionTypeFeature[i,:])

        lesionIndex += numLesions    

    print 'Finding brain types...'
    kmeans = KMeans(n_clusters=numBrainTypes, n_jobs=8)
    kmeans.fit_predict(lesionTypeFeature)
    print 'Done'

    plt.hist(kmeans.labels_)
    plt.show()
    



if __name__ == "__main__":
    main()
#    compareLesionClusterings(5000)