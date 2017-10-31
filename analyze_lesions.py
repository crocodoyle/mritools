import cPickle as pkl
import numpy as np

import csv

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GMM, DPGMM
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.cross_validation import train_test_split

from sklearn.metrics import silhouette_score

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier


from scipy.spatial.distance import euclidean
from scipy import stats

import time

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

import vtk
import nibabel as nib

import os
import collections
from collections import defaultdict

import random
import sys

# these are the modules that I wrote
import context_extraction, load_data
import bol_classifiers

reload(context_extraction)
reload(load_data)
reload(bol_classifiers)

import subprocess


import warnings
warnings.filterwarnings("ignore")


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
    
def cluster(data, numClusters):    
    startTime = time.time()
    kmeans = KMeans(n_clusters=numClusters, n_jobs=threads, copy_x=False)
    kmeans.fit_predict(data)
    endTime = time.time()
    
    elapsed = endTime - startTime
    print "Total time elapsed:", elapsed/60, 'minutes'

    return kmeans

def countLesionTypes(mri_list, clusters, flattenedData, numLesions):
    lesionTypes = np.zeros((len(mri_list), len(clusters)))
    
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
           clusterDistances = np.sum(np.square( clusters - lesion ) )
           lesionTypes[i, np.argmin(clusterDistances)] += 1.0 / float(len(scan.lesionList))

    return lesionTypes
    
    #    for d in data:
#        plt.plot(np.ndarray.flatten(d))
#    
#    plt.title('context')
#    plt.show() 
def ics(data, k, n):
    nCluster = np.shape(data)[0]
    variance = np.var(data)
    dims = np.shape(data)[1]
    
    logLikelihood = -(nCluster/2)*np.log(2*np.pi) - ((nCluster*dims)/2)*np.log(variance) - (nCluster - k)/2 + nCluster*np.log(nCluster) - nCluster*np.log(n)
    
    bic = logLikelihood - k*dims*np.log(n)
    aic = logLikelihood - 2*dims
    return bic, aic


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
    if len(dataVectors) == 5:
        data = np.hstack((dataVectors[0], dataVectors[1], dataVectors[2], dataVectors[3], dataVectors[4]))
    if len(dataVectors) == 6:
        data = np.hstack((dataVectors[0], dataVectors[1], dataVectors[2], dataVectors[3], dataVectors[4], dataVectors[5]))
    
    return data

def getNClosest(candidate, n, allLesionFeatures):

    distance = np.zeros((np.shape(allLesionFeatures)[0]))

    for i, lesionFeatures in enumerate(allLesionFeatures):
        distance[i] = euclidean(candidate, lesionFeatures)
    
    nClosest = distance.argsort()[:n+1]

    return nClosest

def getNClosestMahalanobis(candidate, n, allLesionFeatures):
#    print 'initializing...'
#    sys.stdout.flush()    
    distances = np.zeros(np.shape(allLesionFeatures)[0])
#    print 'calculating variance'
#    sys.stdout.flush() 
    variance = np.var(allLesionFeatures, axis=0)
#    print np.shape(variance)
#    print 'calculated variance'
#    sys.stdout.flush() 
    for i, example in enumerate(allLesionFeatures):
#        print 'distance', i, '/', len(allLesionFeatures)
        distances[i] = np.sum(np.divide((candidate - example), variance)**2)
        sys.stdout.flush()
#    print 'calculated distance'
#    sys.stdout.flush()
    nClosest = distances.argsort()[:n]
#    print 'sorted distances'
#    sys.stdout.flush()
    return nClosest    
    
def selectFeatures(features, experiment, lesionSize):
    totalFeatures = np.shape(features)[1]
    
    mi_scores = np.ones((totalFeatures, totalFeatures))
    
    clusterLabels = np.zeros((np.shape(features)[0], np.shape(features)[1]))
    
    print 'computing labels...'
    for i in range(totalFeatures):
#        print 'computing labels for feature', i, '/', totalFeatures
        cluster = clusterEM(features[:,i], 5)
        clusterLabels[:,i] = cluster.predict(features[:,i])
        
    print 'computing mutual information...'
    for i in range(totalFeatures):
        for j in range(totalFeatures)[i+1:]:
#            print 'computer mutual information for features', i, j
            mi_scores[i,j] = mutual_info_score(clusterLabels[:,i], clusterLabels[:,j])
            mi_scores[j,i] = mi_scores[i,j]
    
    plt.imshow(mi_scores, interpolation='nearest')
    plt.colorbar()
    plt.title('Feature MI for ' + experiment + ' predicting ' + lesionSize + ' lesions')
    plt.savefig('/usr/local/data/adoyle/images/mi-' + experiment + '-' + lesionSize + '.png', dpi=100)
#    plt.show()     

def clusterEM(data, numClusters):
    clusterer = GMM(n_components = numClusters, covariance_type='full')
    c = clusterer.fit(data)

    return c

def createRepresentationSpace(mri_list, dataVectors, lesionSizes, numWithClinical, lesionCentroids, examineClusters=False):
    subtypeShape = []

    clusters = []
    lesionTypes = []
    
    brainIndices = {}
    lesionIndices = {}
    
    brainsOfType = {}
    lesionsOfType = {}
    
    for m, size in enumerate(sizes):
        brainIndices[size] = defaultdict(list)
        lesionIndices[size] = defaultdict(list)
        
        brainsOfType[size] = defaultdict(list)
        lesionsOfType[size] = defaultdict(list)

    for m, size in enumerate(sizes):
        subtypeShape.append( () )
        subtypeShape[m] += (len(mri_list),)
        
        clusterAssignments = []
        clusterProbabilities = []
        clusters.append([])

        for d, data in enumerate(dataVectors):
            lesionFeatures = data[size]
            print "START OF", sizes[m], feats[d]
            print np.shape(lesionFeatures)
   
            numClusters = []
            bics = []
            aics = []
            scores = []
            silhouette = []
            clustSearch = []
            clustSearch.append("")
            clustSearch.append("")
            
#            if feats[d] != 'Intensity':
            clusterData, validationData = train_test_split(lesionFeatures, test_size=0.3, random_state=5)
            for k in range(2,4):
                print 'trying ' + str(k) + ' clusters...'
                clustSearch.append(GMM(n_components = k, covariance_type = 'full'))
                clustSearch[k].fit(clusterData)
                            
                numClusters.append(k)
                bics.append(clustSearch[k].bic(validationData))
                aics.append(clustSearch[k].aic(validationData))
                scores.append(np.mean(clustSearch[k].score(validationData)))
                
#                print np.shape(validationData)
#                sil = silhouette_score(validationData, clustSearch[k].predict(validationData), metric='mahalanobis')             
#                silhouette.append(sil)
                
            nClusters = numClusters[np.argmin(bics)]
#            nClusters = numClusters[np.argmax(silhouette)]
            
#            fig, (ax, ax2) = plt.subplots(1,2, figsize=(12, 4))
#            
#            ax.plot(numClusters, bics, label="BIC")
#            ax.plot(numClusters, aics, label="AIC")
#            ax.set_xlabel("# of " + feats[d] + " sub-types of " + sizes[m] + " lesions")
#            ax.set_ylabel("Information Criterion (lower is better)")
#            ax.legend()
#
#            ax2.plot(numClusters, scores, label="Log Prob")
#            ax2.set_xlabel("# of " + feats[d] + " sub-types of " + sizes[m] + " lesions")
#            ax2.set_ylabel("Average Log Probability (higher is better)")
#            ax2.legend()
#            
##            ax3.plot(numClusters, silhouette, label="Avg. Silhouette")
##            ax3.set_xlabel("# of " + feats[d] + " sub-types of " + sizes[m] + " lesions")
##            ax3.set_ylabel("Average Silhouette (higher is better)")
##            ax3.legend()
##            
#            plt.tight_layout()
#            plt.show()
#            plt.close()
                
#            else:
#                nClusters = 5
            
            print "Selected " + str(nClusters) + " clusters for " + feats[d] + " in " + sizes[m] + " lesions"
            sys.stdout.flush()
            
            c = GMM(n_components = nClusters, covariance_type = 'full')
            c.fit(lesionFeatures)
            
            subtypeShape[m] += (nClusters, )
        
            clusterAssignments.append(c.predict(lesionFeatures))
            clusterProbabilities.append(c.predict_proba(lesionFeatures))
            clusters[m].append(c)
            
            
#            fig, (ax, ax2, ax3) = plt.subplots(1,3, figsize=(12,4))            
#            
#            #visualize lesion sub-type distribution
#            hist, bins = np.histogram(clusterAssignments[-1], nClusters)
#            ax.plot(np.add(range(nClusters),1), hist)
#            ax.set_xlabel(feats[d] + ' ML sub-type lesion distribution')
#            ax.set_ylabel('number of ' + sizes[m] + ' lesions')
#
#            entropies = []
#            for l in lesionFeatures:
#                if not np.isnan(stats.entropy(l)).any():
#                    entropies.append(stats.entropy(l))
#                
#            hist, bins = np.histogram(entropies)
#            ax2.plot(hist)
#            ax2.set_xlabel('entropy value for ' + feats[d] + ' features')
#            ax2.set_ylabel('number of ' + sizes[m] + ' lesions')
#        
#            hist, bins = np.histogram(lesionFeatures, 100)
#            ax3.plot(hist, label=sizes[m] + " - " + feats[d])
#            ax3.set_xlabel("value of feature")
#            ax3.set_ylabel("number of lesions")
#            ax3.legend()
#            
#            plt.tight_layout()
#            plt.show() 
#            plt.close()
        
        
        lesionTypes.append(np.zeros(subtypeShape[m]))
        
#        randomLesionType = (np.random.randint(shape[1]), np.random.randint(shape[2]), np.random.randint(shape[3]), np.random.randint(shape[4]), m)

        print "Subtypes for " + sizes[m] + ": ", subtypeShape[m] 

        print "Combining lesion subtypes..."
        lesionIndex = 0
        for i, scan in enumerate(mri_list):
            for j, lesion in enumerate(scan.lesionList):
                if (len(lesion) > 2 and len(lesion) < 11 and m == 0) or (len(lesion) > 10 and len(lesion) < 26 and m == 1) or (len(lesion) > 25 and len(lesion) < 101 and m == 2) or (len(lesion) > 100 and m == 3):     

                    for f1 in range(subtypeShape[m][1]):
                        for f2 in range(subtypeShape[m][2]):
                            for f3 in range(subtypeShape[m][3]):
                                for f4 in range(subtypeShape[m][4]):
                                    lesionTypes[m][i, f1, f2, f3, f4] += clusterProbabilities[0][lesionIndex, f1]*clusterProbabilities[1][lesionIndex, f2]*clusterProbabilities[2][lesionIndex, f3]*clusterProbabilities[3][lesionIndex, f4]

                    brainIndices[size][''.join((str(clusterAssignments[0][lesionIndex]),str(clusterAssignments[1][lesionIndex]),str(clusterAssignments[2][lesionIndex]),str(clusterAssignments[3][lesionIndex])))].append(i)
                    lesionIndices[size][''.join((str(clusterAssignments[0][lesionIndex]),str(clusterAssignments[1][lesionIndex]),str(clusterAssignments[2][lesionIndex]),str(clusterAssignments[3][lesionIndex])))].append(j)

                    lesionIndex += 1


        if visualizeAGroup:
            n = 6
            
            for f1 in range(subtypeShape[m][1]):
                for f2 in range(subtypeShape[m][2]):
                    for f3 in range(subtypeShape[m][3]):
                        for f4 in range(subtypeShape[m][4]):
                            lesionToViz = ''.join((str(f1),str(f2),str(f3),str(f4)))
                            plt.figure(figsize=(8.5,2.5))
            

                            for i, (brainIndex, lesionIndex) in enumerate(zip(brainIndices[size][lesionToViz][0:n], lesionIndices[size][lesionToViz][0:n])):
                                scan = mri_list[brainIndex]
#                                    img = nib.load(scan.images['flr']).get_data()
                                img = nib.load(scan.images['t2w']).get_data()
                                lesionMaskImg = np.zeros((np.shape(img)))
                                
#                                for lesion in scan.lesionList:
#                                    for point in lesion:
#                                        lesionMaskImg[point[0], point[1], point[2]] = 1
                                
                                for point in scan.lesionList[lesionIndex]:
                                    lesionMaskImg[point[0], point[1], point[2]] = 1
                                
                                x, y, z = [int(np.mean(xxx)) for xxx in zip(*scan.lesionList[lesionIndex])]
                    
                                maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)                      
                                maskSquare = np.zeros((np.shape(img)))
                                maskSquare[x-10:x+10, y+10, z] = 1
                                maskSquare[x-10:x+10, y-10, z] = 1
                                maskSquare[x-10, y-10:y+10, z] = 1
                                maskSquare[x+10, y-10:y+10, z] = 1
                                              
                                square = np.ma.masked_where(maskSquare == 0, np.ones(np.shape(maskSquare))*5000)
                       
                                lesionMaskPatch = maskImg[x-20:x+20, y-20:y+20, z]
                                ax = plt.subplot(2, n, i+1)
                                ax.axis('off')
                                ax.imshow(img[20:200,20:200, z].T, cmap = plt.cm.gray, interpolation = 'nearest',origin='lower')
                                ax.imshow(maskImg[20:200,20:200, z].T, cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
                                ax.imshow(square[20:200, 20:200, z].T, cmap = plt.cm.autumn, interpolation = 'nearest', origin='lower')
                                                
                                ax3 = plt.subplot(2, n, i+1+n)
                                ax3.imshow(img[x-20:x+20, y-20:y+20, z].T, cmap = plt.cm.gray, interpolation = 'nearest', origin='lower')
                                ax3.imshow(lesionMaskPatch.T, cmap = plt.cm.autumn, alpha = 0.4, interpolation = 'nearest', origin='lower')
                                ax3.axes.get_yaxis().set_visible(False)
                                ax3.set_xticks([])
                                ax3.set_xlabel(letters[i])
                               
#                                title = 'Examples of Same Lesion-Type'
#                                plt.suptitle(title, fontsize=32)
                            plt.subplots_adjust(wspace=0.01,hspace=0.01)
                            plt.savefig('/usr/local/data/adoyle/images/t2lesions-'+ size + '-' + ''.join((str(f1),str(f2),str(f3),str(f4))) + '.png', dpi=500)
#                            plt.show()


    pcas = []
    lesionFlat = []
    if usePCA:
        print "applying PCA..."
        
        pcaTransformedData = []
        for m, size in enumerate(sizes):
            lesionBins = 1
            for dims in np.shape(lesionTypes[m])[1:]:
                lesionBins *= dims
    
            lesionFlat = np.reshape(lesionTypes[m], (len(mri_list), lesionBins))
            pcas.append(PCA(n_components = 0.95).fit(lesionFlat))
            pcaTransformedData.append(pcas[-1].transform(lesionFlat))
            
        data = np.hstack((pcaTransformedData[0], pcaTransformedData[1], pcaTransformedData[2], pcaTransformedData[3]))
    else:
        for m, size in enumerate(sizes):
            lesionBins = 1
            for dims in np.shape(lesionTypes[m])[1:]:
                lesionBins *= dims
        
            lesionFlat.append(np.reshape(lesionTypes[m], (len(mri_list), lesionBins)))
            
        data = np.hstack((lesionFlat[0], lesionFlat[1], lesionFlat[2], lesionFlat[3]))


    data = data[:, 0:numWithClinical, ...]
    
    
    for m, size in enumerate(sizes):
        lesionType = 0
        for f1 in range(subtypeShape[m][1]):
            for f2 in range(subtypeShape[m][2]):
                for f3 in range(subtypeShape[m][3]):
                    for f4 in range(subtypeShape[m][4]):
                        brainsOfType[size][lesionType] = brainIndices[size][''.join((str(f1),str(f2),str(f3),str(f4)))]
                        lesionsOfType[size][lesionType] = lesionIndices[size][''.join((str(f1),str(f2),str(f3),str(f4)))]

                        lesionType += 1

    return data, clusters, pcas, subtypeShape, brainsOfType, lesionsOfType

def testRepresentationSpace(mri_list, dataVectors, lesionSizes, clusters, pcas):
    subtypeShape = []
    lesionTypes = []


    for m, size in enumerate(sizes):
#        clusterAssignments = []
        clusterProbabilities = []
        
        subtypeShape.append( () )
        
        subtypeShape[m] += (len(mri_list),)
        subtypeShape[m] += tuple(c.n_components for c in clusters[m])
        
        lesionTypes.append(np.zeros(subtypeShape[m]))

        for d, data in enumerate(dataVectors):
            lesionFeatures = data[size]
            c = clusters[m][d]
            
#            clusterAssignments.append(c.predict(lesionFeatures))
            clusterProbabilities.append(c.predict_proba(lesionFeatures))

        lesionIndex = 0
        for i, scan in enumerate(mri_list):
            for j, lesion in enumerate(scan.lesionList):
                if (len(lesion) > 2 and len(lesion) < 11 and m == 0) or (len(lesion) > 10 and len(lesion) < 26 and m == 1) or (len(lesion) > 25 and len(lesion) < 101 and m == 2) or (len(lesion) > 100 and m == 3):             
                    for f1 in range(subtypeShape[m][1]):
                        for f2 in range(subtypeShape[m][2]):
                            for f3 in range(subtypeShape[m][3]):
                                for f4 in range(subtypeShape[m][4]):
                                    lesionTypes[m][i, f1, f2, f3, f4] += clusterProbabilities[0][lesionIndex, f1]*clusterProbabilities[1][lesionIndex, f2]*clusterProbabilities[2][lesionIndex, f3]*clusterProbabilities[3][lesionIndex, f4]
                    lesionIndex += 1
                    
    lesionFlat = []
    if usePCA:
        pcaTransformedData = []
        for m, size in enumerate(sizes):
            lesionBins = 1
            for dims in np.shape(lesionTypes[m])[1:]:
                lesionBins *= dims
    
            lesionFlat = np.reshape(lesionTypes[m], (len(mri_list), lesionBins))
            pcaTransformedData.append(pcas[m].transform(lesionFlat))
            
        data = np.hstack((pcaTransformedData[0], pcaTransformedData[1], pcaTransformedData[2], pcaTransformedData[3]))
    else:
        for m, size in enumerate(sizes):
            lesionBins = 1
            for dims in np.shape(lesionTypes[m])[1:]:
                lesionBins *= dims
        
            lesionFlat.append(np.reshape(lesionTypes[m], (len(mri_list), lesionBins)))
            
        data = np.hstack((lesionFlat[0], lesionFlat[1], lesionFlat[2], lesionFlat[3]))
        
        
    return data

def analyzeClinical(mri_list, clusterAssignments):
    patientIndex = {}
    atrophy = {}
    newT1 = {}
    newT2 = {}
    
    for treatment in treatments:
        patientIndex[treatment] = 0
        atrophy[treatment] = []
        newT1[treatment] = []
        newT2[treatment] = []    
        for i in range(len(set(clusterAssignments['Placebo']))):
            atrophy[treatment].append([])
            newT1[treatment].append([])
            newT2[treatment].append([])

        
    for i, scan in enumerate(mri_list):
        atrophy[scan.treatment][clusterAssignments[scan.treatment][patientIndex[scan.treatment]]].append(scan.atrophy)
        newT1[scan.treatment][clusterAssignments[scan.treatment][patientIndex[scan.treatment]]].append(scan.newT1)
        newT2[scan.treatment][clusterAssignments[scan.treatment][patientIndex[scan.treatment]]].append(scan.newT2)
        patientIndex[scan.treatment] += 1
        
    
#    fig, axes = plt.subplots(nrows=len(treatments), ncols=1, figsize=(12, 8))
#    fig2, axes2 = plt.subplots(nrows=len(treatments), ncols=1, figsize=(12, 8))
    fig3, axes3 = plt.subplots(nrows=len(treatments), ncols=1, figsize=(12, 8))

    for tr, treatment in enumerate(treatments):
#        data = [atrophy[treatment][0], atrophy[treatment][1], atrophy[treatment][2], atrophy[treatment][3], atrophy[treatment][4], atrophy[treatment][5], atrophy[treatment][6], atrophy[treatment][7], atrophy[treatment][8], atrophy[treatment][9], atrophy[treatment][10]]
#        data2 = [newT1[treatment][0], newT1[treatment][1], newT1[treatment][2], newT1[treatment][3], newT1[treatment][4], newT1[treatment][5], newT1[treatment][6], newT1[treatment][7], newT1[treatment][8], newT1[treatment][9], newT1[treatment][10]]
        data3 = [newT2[treatment][0], newT2[treatment][1], newT2[treatment][2], newT2[treatment][3], newT2[treatment][4], newT2[treatment][5], newT2[treatment][6], newT2[treatment][7], newT2[treatment][8], newT2[treatment][9], newT2[treatment][10]]

#        axes[tr].boxplot(data)
#        axes[tr].set_ylabel(treatment, fontsize=14)
#        axes2[tr].boxplot(data2)
#        axes2[tr].set_ylabel(treatment, fontsize=14)
        axes3[tr].boxplot(data3)
        axes3[tr].set_ylabel(treatment, fontsize=14)
    
    plt.suptitle('New T2 Lesions by Patient Groups', fontsize=24)
    plt.xlabel('Brain Cluster', fontsize=14)
#    plt.tight_layout()
#    plt.show() 
    plt.close('all')
        
        

def analyzeClinical2(mri_list, clusterAssignments, groupProbabilities):
    atrophy = {}
    newT1 = {}
    newT2 = {}
    
    nClusters = np.shape(groupProbabilities)[1]
    
    for treatment in treatments:
        atrophy[treatment] = []
        newT1[treatment] = []
        newT2[treatment] = []    
        for i in range(nClusters):
            atrophy[treatment].append([])
            newT1[treatment].append([])
            newT2[treatment].append([])


    for i, scan in enumerate(mri_list):
        atrophy[scan.treatment][clusterAssignments[i]].append(scan.atrophy)
        newT1[scan.treatment][clusterAssignments[i]].append(scan.newT1)
        newT2[scan.treatment][clusterAssignments[i]].append(scan.newT2)
        
    
    fig, axes = plt.subplots(nrows=len(treatments), ncols=1, figsize=(12, 8), sharey=True)
    fig2, axes2 = plt.subplots(nrows=len(treatments), ncols=1, figsize=(12, 8), sharey=True)
    fig3, axes3 = plt.subplots(nrows=len(treatments), ncols=1, figsize=(12, 8), sharey=True)

    for tr, treatment in enumerate(treatments):
        axes[tr].boxplot(atrophy[treatment])
        axes[tr].set_ylabel(treatment, fontsize=14)
        axes2[tr].boxplot(newT1[treatment])
        axes2[tr].set_ylabel(treatment, fontsize=14)
        axes3[tr].boxplot(newT2[treatment])
        axes3[tr].set_ylabel(treatment, fontsize=14)

    for tr, treatment in enumerate(treatments):
        top = axes[tr].get_ylim()[1]
        pos = np.arange(nClusters) + 1
        
        for tick, label in zip(range(nClusters), axes[tr].get_xticklabels()):
            axes[tr].text(pos[tick], top - (top*0.2), 'n='+str(len(atrophy[treatment][tick])), horizontalalignment='center', size='small')
            axes[tr].text(pos[tick], top - (top*0.4), 'mean='+str(np.round(np.mean(atrophy[treatment][tick]),1)), horizontalalignment='center', size='small')
        
        top = axes2[tr].get_ylim()[1]
        for tick, label in zip(range(nClusters), axes2[tr].get_xticklabels()):
            axes2[tr].text(pos[tick], top - (top*0.1), 'n='+str(len(newT1[treatment][tick])), horizontalalignment='center', size='small')
            axes2[tr].text(pos[tick], top - (top*0.2), 'mean='+str(np.round(np.mean(newT1[treatment][tick]), 1)), horizontalalignment='center', size='small')
        
        top = axes3[tr].get_ylim()[1]
        for tick, label in zip(range(nClusters), axes3[tr].get_xticklabels()):
            axes3[tr].text(pos[tick], top - (top*0.1), 'n='+str(len(newT2[treatment][tick])), horizontalalignment='center', size='small')
            axes3[tr].text(pos[tick], top - (top*0.2), 'mean='+str(np.round(np.mean(newT2[treatment][tick]), 1)), horizontalalignment='center', size='small')

    
    print 'showing figures'
    fig.suptitle('Atrophy by Patient Groups', fontsize=24)
    fig.savefig('/usr/local/data/adoyle/images/atrophy.png')
    fig.show()

    fig2.suptitle('New T1 Lesions by Patient Groups', fontsize=24)
    fig2.savefig('/usr/local/data/adoyle/images/newt1.png')
    fig2.show()

    fig3.suptitle('New T2 Lesions by Patient Groups', fontsize=24)
    fig3.savefig('/usr/local/data/adoyle/images/newt2.png')
    fig3.show()
    plt.show() 
 
def analyzeClinical3(mri_list, clusterAssignments, groupProbabilities):
    atrophy = {}
    newT1 = {}
    newT2 = {}
    
    nClusters = np.shape(groupProbabilities)[1]
    
    for treatment in treatments:
        atrophy[treatment] = []
        newT1[treatment] = []
        newT2[treatment] = []    
        for i in range(nClusters):
            atrophy[treatment].append([])
            newT1[treatment].append([])
            newT2[treatment].append([])

    for i, scan in enumerate(mri_list):
        atrophy[scan.treatment][clusterAssignments[i]].append(scan.atrophy)
        newT1[scan.treatment][clusterAssignments[i]].append(scan.newT1)
        newT2[scan.treatment][clusterAssignments[i]].append(scan.newT2)

    #remove empty clusters
    i = 0
    while i < len(atrophy[treatment]):
        n=0
        for treatment in treatments:
            n += len(atrophy[treatment][i])
        if n == 0:
            for treatment in treatments:
                del atrophy[treatment][i]
                del newT1[treatment][i]
                del newT2[treatment][i]
            i -= 1
        i += 1
    
    nClusters = len(atrophy[treatment])
    
    plt.close()
    fig, axes = plt.subplots(nrows=len(treatments), ncols=1, figsize=(12, 8), sharey=True)
    fig2, axes2 = plt.subplots(nrows=len(treatments), ncols=1, figsize=(12, 8), sharey=True)
    fig3, axes3 = plt.subplots(nrows=len(treatments), ncols=1, figsize=(12, 8), sharey=True)

    xticks = np.linspace(1, nClusters, num=nClusters)


    epsilon = 1e-7
    t1_active = np.zeros((len(treatments), nClusters))
    t1_nonactive = np.zeros((len(treatments), nClusters))
    
    t2_active = np.zeros((len(treatments), nClusters))
    t2_nonactive = np.zeros((len(treatments), nClusters))

    bar_width = 0.35
    
    print np.shape(xticks)
    
    for tr, treatment in enumerate(treatments):
        for j in range(nClusters):
            t1_active[tr, j] = np.count_nonzero(newT1[treatment][j]) + epsilon
            t1_nonactive[tr, j] = len(newT1[treatment][j]) - np.count_nonzero(newT1[treatment][j]) + epsilon
            
            t2_active[tr, j] = np.count_nonzero(newT2[treatment][j]) + epsilon
            t2_nonactive[tr, j] = len(newT2[treatment][j]) - np.count_nonzero(newT2[treatment][j]) + epsilon
            
        axes[tr].bar(xticks-0.2, t1_active[tr, :], bar_width, color='r')
        axes[tr].bar(xticks, t1_nonactive[tr, :], bar_width, color='g')
        axes[tr].set_ylabel(treatment, fontsize=14)

        axes2[tr].bar(xticks-0.2, t2_active[tr, :], bar_width, color='r')
        axes2[tr].bar(xticks, t2_nonactive[tr, :], bar_width, color='g')
        axes2[tr].set_ylabel(treatment, fontsize=14)
            
    for tr, treatment in enumerate(treatments):
        top = axes[tr].get_ylim()[1]
        pos = np.arange(nClusters) + 1
        
        for tick, label in zip(range(nClusters), axes[tr].get_xticklabels()):
            axes[tr].text(pos[tick], top - (top*0.1), 'n='+str(len(newT1[treatment][tick])), horizontalalignment='center', size='small')
            axes[tr].text(pos[tick], top - (top*0.2), 'mean='+str(np.round(np.mean(newT1[treatment][tick]),1)), horizontalalignment='center', size='small')
        
        top = axes2[tr].get_ylim()[1]
        for tick, label in zip(range(nClusters), axes2[tr].get_xticklabels()):
            axes2[tr].text(pos[tick], top - (top*0.1), 'n='+str(len(newT2[treatment][tick])), horizontalalignment='center', size='small')
            axes2[tr].text(pos[tick], top - (top*0.2), 'mean='+str(np.round(np.mean(newT2[treatment][tick]), 1)), horizontalalignment='center', size='small')
    
    print 'showing figures'
    fig.suptitle('New T1 Lesions by Patient Groups', fontsize=24)
    fig.savefig('/usr/local/data/adoyle/images/newt1.png')
    fig.show()

    fig2.suptitle('New T2 Lesions by Patient Groups', fontsize=24)
    fig2.savefig('/usr/local/data/adoyle/images/newt2.png')
    fig2.show()
    plt.show()     
 
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
    ax.set_xticks([])
#    ax.set_xticks(xticks)
#    ax.set_xticklabels(x)
    
    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True)
    
#    for tick in ax.xaxis.get_major_ticks():
#        tick.label.set_fontsize(6)
#        tick.label.set_rotation('vertical')

    scale = 1. / ly
    for pos in xrange(ly + 1):
        add_line(ax, pos * scale, -.1)
    ypos = -.2
    while groups:
        group = groups.pop()
        pos = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            if label=="LES" or label=='T' or label=="M" or label=="S" or label=="L":
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

def brainStatistics():
    mri_list = pkl.load(open('/usr/local/data/adoyle/mri_list.pkl', 'rb'))

    lesions = np.zeros((len(mri_list), 4))
    numLesions = np.zeros(len(mri_list))
    
    for i, scan in enumerate(mri_list):
        for lesion in scan.lesionList:
            lesionSize = len(lesion)
            if lesionSize <= 10 and lesionSize > 2:
                lesions[i, 0] += 1
                numLesions[i] += 1
            if lesionSize > 10 and lesionSize <= 25:
                lesions[i, 1] += 1
                numLesions[i] += 1
            if lesionSize > 25 and lesionSize <= 100:
                lesions[i, 2] += 1
                numLesions[i] += 1
            if lesionSize > 100:
                lesions[i, 3] += 1
                numLesions[i] += 1
                
    plt.hist(lesions[:,0], 30)
    plt.title('tiny lesions histogram')
    plt.ylabel('total cases')
    plt.xlabel('number of lesions')
    plt.show() 
    
    plt.hist(lesions[:,1], 30)
    plt.title('small lesions histogram')
    plt.ylabel('total cases')
    plt.xlabel('number of lesions')
    plt.show() 
    
    plt.hist(lesions[:,2], 30)
    plt.title('medium lesions histogram')
    plt.ylabel('total cases')
    plt.xlabel('number of lesions')
    plt.show() 
    
    plt.hist(lesions[:,3], 20)
    plt.title('big lesions histogram')
    plt.ylabel('total cases')
    plt.xlabel('number of lesions')
    plt.show() 
    
    plt.hist(numLesions, 30)
    plt.title('total lesions histogram')
    plt.ylabel('total cases')
    plt.xlabel('number of lesions')
    plt.show() 
    
    
    
    hist = np.histogram2d(lesions[:,0], lesions[:, 1], bins = 40)
    
    plt.imshow(hist[0][:15, :15], interpolation='nearest', cmap = plt.cm.gray, origin='lower')
    plt.title('joint histogram, tiny and small')
    plt.xlabel('tiny')
    plt.ylabel('small')
    plt.colorbar()
    plt.show() 
    
    hist = np.histogram2d(lesions[:,0], lesions[:, 2], bins = 40)
    plt.imshow(hist[0][:15, :15], interpolation='nearest', cmap = plt.cm.gray, origin='lower')
    plt.title('joint histogram, tiny and medium')
    plt.xlabel('tiny')
    plt.ylabel('medium')
    plt.colorbar()
    plt.show()     
    
    hist = np.histogram2d(lesions[:,0], lesions[:, 3], bins = 30)
    plt.imshow(hist[0][:15, :15], interpolation='nearest', cmap = plt.cm.gray, origin='lower')
    plt.title('joint histogram, tiny and large')
    plt.xlabel('tiny')
    plt.ylabel('large')
    plt.colorbar()
    plt.show() 
    
    hist = np.histogram2d(lesions[:,1], lesions[:, 2], bins = 30)
    plt.imshow(hist[0][:15, :15], interpolation='nearest', cmap = plt.cm.gray, origin='lower')
    plt.title('joint histogram, small and medium')
    plt.xlabel('small')
    plt.ylabel('medium')
    plt.colorbar()
    plt.show() 
    
    hist = np.histogram2d(lesions[:,1], lesions[:, 3], bins = 30)
    plt.imshow(hist[0][:15, :15], interpolation='nearest', cmap = plt.cm.gray, origin='lower')
    plt.title('joint histogram, small and large')
    plt.xlabel('small')
    plt.ylabel('large')
    plt.colorbar()
    plt.show() 
    

    hist = np.histogram2d(lesions[:,2], lesions[:, 3], bins = 30)
    plt.imshow(hist[0][:15, :15], interpolation='nearest', cmap = plt.cm.gray, origin='lower')
    plt.title('joint histogram, medium and large')
    plt.ylabel('large')
    plt.xlabel('medium')
    plt.colorbar()
    plt.show()   
    
    
    if selectK:
        numClusters = []
        bics = []
        for i in range(3, 30):
            c = clusterEM(lesions, i)
            numClusters.append(i)
            bic = c.bic(lesions)
            bics.append(bic)
                
            print 'optimal clusters:', numClusters[np.argmin(bics)]
            print numClusters
                    
        plt.plot(numClusters, bics)
        nClusters = numClusters[np.argmin(bics)]
    else:
        nClusters = 13
    

    c = clusterEM(lesions, nClusters)
    clusterAssignments = c.predict(lesions)
    
    queryBrainIndices = np.zeros(nClusters)
    
    if os.path.exists('/usr/local/data/adoyle/brainIndices3.pkl'):
        queryBrainIndices = pkl.load(open('/usr/local/data/adoyle/brainIndices3.pkl'))
    else:
        for i in range(nClusters):
            query = np.random.randint(len(mri_list))
            while clusterAssignments[query] != i:
                query = np.random.randint(len(mri_list))
            queryBrainIndices[i] = query
                

    for brainCluster in range(nClusters):
        nClosest = getNClosest(lesions[queryBrainIndices[brainCluster], :], 6, lesions)
        
        print nClosest
        
        globalymax = 0
        fig = plt.figure(figsize=(12,4))
        axes = []       
        for n, goodOne in enumerate(nClosest):
            scan = mri_list[goodOne]            
            t2 = nib.load(mri_list[goodOne].images['t2w']).get_data()
            
            lesionMaskImg = np.zeros((np.shape(t2)))
            
            for lesion in scan.lesionList:
                for point in lesion:
                    lesionMaskImg[point[0], point[1], point[2]] = 1
                            
            maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)                      

            ax = fig.add_subplot(2, 7, n+1)
            ax.imshow(t2[20:200, 20:200, 30].T, cmap=plt.cm.gray, origin='lower')
            ax.imshow(maskImg[20:200,20:200, 30].T, cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
            ax.axis('off')
            if n == 0:
                ax.set_title('query')
            ax.set_xlabel(clusterAssignments[goodOne])


            if n==0:
                axes.append(fig.add_subplot(2, 7, n+8))
            else:
                axes.append(fig.add_subplot(2, 7, n+8, sharey=axes[0]))
                axes[n].axes.get_yaxis().set_visible(False)
                
            ind = np.arange(4) + 0.5
            axes[n].bar(ind, lesions[goodOne, :], 0.5)
            axes[n].set_xticks(ind)
            axes[n].set_xticklabels(('T', 'S', 'M', 'L'))
#            ax.axes.get_yaxis().set_visible(False)
            
            ymax = np.max(lesions[goodOne, :])
            if ymax > globalymax:
                globalymax = ymax
            axes[n].set_ylim([0, globalymax])

        plt.subplots_adjust(wspace=0.01,hspace=0.01)
        plt.suptitle('brains from group ' + str(brainCluster+1), fontsize=20)
        plt.savefig('/usr/local/data/adoyle/images/brains' + str(brainCluster) + '.png', dpi=500)
                
        plt.show() 
        
    pkl.dump(queryBrainIndices, open('/usr/local/data/adoyle/brainIndices3.pkl', 'wb'))


def testFeatures(mri_list, lesionTypes):
    lr = LinearRegression()
    
    atrophy = np.zeros((len(mri_list), 1))    
    newT1 = np.zeros((len(mri_list), 1))
    newT2 = np.zeros((len(mri_list), 1))   
    
    for i, scan in enumerate(mri_list):
        atrophy[i] = scan.atrophy
        newT1[i] = scan.newT1
        newT2[i] = scan.newT2
        
    scores = {}
    correlation = {}
    for metric in metrics:
        scores[metric] = []
        correlation[metric] = []

    for lesionType in range(np.shape(lesionTypes)[1]):
        x = np.reshape(lesionTypes[:, lesionType], (len(mri_list), 1))
        
        correlation['atrophy'].append(np.abs(stats.pearsonr(x,atrophy)[0]))      
        correlation['newT1'].append(stats.pearsonr(x,newT1)[0])
        correlation['newT2'].append(stats.pearsonr(x,newT2)[0])
        
        lr.fit(x, atrophy)
        scores['atrophy'].append(lr.score(x, atrophy))
        
        lr.fit(x, newT1)
        scores['newT1'].append(lr.score(x, newT1))
        
        lr.fit(x, newT2)
        scores['newT2'].append(lr.score(x, newT2))

    plt.plot(scores['atrophy'], label='atrophy')
    plt.plot(scores['newT1'], label='newT1')
    plt.plot(scores['newT2'], label='newT2')
    
    plt.title('Feature scores predicting clinical outcomes')
    plt.xlabel('Feature')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    
    plt.plot(correlation['atrophy'], label='atrophy')
    plt.plot(correlation['newT1'], label='newT1')
    plt.plot(correlation['newT2'], label='newT2')
    plt.title('Correlation of features and clinical outcomes')
    plt.xlabel('Feature')
    plt.ylabel('Pearson Correlation')
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()



def getOutcomes(mri_list):
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

def visualizePatientGroupHistograms(trainData, trainClusterAssignments):

    fig, axs = plt.subplots(len(set(trainClusterAssignments)), 1, sharey=True, figsize = (32, 3*len(set(trainClusterAssignments))))
    for group in set(trainClusterAssignments):
        patientsInGroup = []
        
        for i, (patientHistogram, clusterNum) in enumerate(zip(trainData, trainClusterAssignments)):
            if clusterNum == group:
                patientsInGroup.append(patientHistogram)
        
        print np.shape(np.asarray(patientsInGroup).T)
        axs[group].boxplot(np.asarray(patientsInGroup), 1, '')
        axs[group].set_title('Group ' + str(group) + ' histogram')
        axs[group].set_xticks([])
        axs[group].set_yticks([])
        axs[group].set_xlabel('Lesion Type')
        
    plt.savefig('/usr/local/data/adoyle/images/groupHistograms.png', dpi=200)
    plt.show()
    


def plotTreatmentHists(mri_list, outcomes):
    data = {}
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12,3))
    for m, metric in enumerate(metrics):    
        
        data[metric] = collections.OrderedDict()
    
        for treatment in treatments:
            data[metric][treatment] = collections.OrderedDict()
            data[metric][treatment]['active'] = 0 
            data[metric][treatment]['non-active'] = 0
    
    
        for i, (scan, outcome) in enumerate(zip(mri_list, outcomes[metric])):
            for treatment in treatments:
                if scan.treatment == treatment:
                    if outcome == 1:
                        data[metric][treatment]['active'] += 1
                    else:
                        data[metric][treatment]['non-active'] += 1
                   
        label_group_bar(axs[m], data[metric])
        axs[m].set_title(metric)

    plt.show()




def visualizePatientGroups(mri_list, trainData, groups, subtypeShape):
    plt.close()
    
    lesionSizeFeatures = {}
    
    for index, l in enumerate(['T', 'S', 'M', 'L']):
        lesionSizeFeatures[l] = subtypeShape[index][1]*subtypeShape[index][2]*subtypeShape[index][3]*subtypeShape[index][4]
    
    for g in set(groups):
        fig = plt.figure(figsize=(15,6), dpi=500)
        n=0
        ymax = 0
        axes = []
        for i, (scan, hist, group) in enumerate(zip(mri_list, trainData, groups)):
            hist = np.add(hist, 0.01)
            if group == g:
                n+=1
                t2 = nib.load(scan.images['t2w']).get_data()
            
                lesionMaskImg = np.zeros((np.shape(t2)))
            
                for lesion in scan.lesionList:
                    for point in lesion:
                        lesionMaskImg[point[0], point[1], point[2]] = 1
                            
                maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)                      
    
                ax = fig.add_subplot(2, 6, n)
                ax.imshow(t2[20:200, 20:200, 30].T, cmap=plt.cm.gray, origin='lower')
                ax.imshow(maskImg[20:200,20:200, 30].T, cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
                ax.axis('off')                

#                axes[-1].bar(range(np.shape(hist)[0]), hist)
                
                ax2 = fig.add_subplot(2, 6, n+6)
                ax2.bar(range(len(hist)), hist, width=1)
                ax2.set_xticks([0, lesionSizeFeatures['T']/2, lesionSizeFeatures['T'], lesionSizeFeatures['T']+(lesionSizeFeatures['S']/2), lesionSizeFeatures['T']+lesionSizeFeatures['S'], lesionSizeFeatures['T']+lesionSizeFeatures['S']+lesionSizeFeatures['M']/2, lesionSizeFeatures['T']+lesionSizeFeatures['S']+lesionSizeFeatures['M'], lesionSizeFeatures['T']+lesionSizeFeatures['S']+lesionSizeFeatures['M']+lesionSizeFeatures['L']/2, lesionSizeFeatures['T']+lesionSizeFeatures['S']+lesionSizeFeatures['M']+lesionSizeFeatures['L']])
                ax2.set_xticklabels(['', 'T', '', 'S', '', 'M', '', 'L', ''])
                
              

#                axes[-1].set_xticks([])
                if np.amax(hist) > ymax:
                    ymax = np.amax(hist)
                
            if n == 6:
                for ax2 in axes:
                    ax2.set_ylim([0, ymax])
                break
            
        groupNum = random.randint(0,100)
        print groupNum
        plt.subplots_adjust(wspace=0.01,hspace=0.01)
        plt.savefig('/usr/local/data/adoyle/images/patient-groups' + str(groupNum) + '-' + str(g) + '.png', dpi=500)       
#        plt.show()
        plt.close()
        
def visualizeWhereTreatmentInfoHelps(example, mri_test, testData, mri_train, trainData):
    print 'example', example
    print 'test data', testData[example, :]
    sys.stdout.flush()
    closeOnes = getNClosestMahalanobis(testData[example, ...], 20, trainData)
    print 'found closest ones:', closeOnes
    sys.stdout.flush()
    visualize = []
    for index in closeOnes:
        if mri_train[index].treatment == 'Avonex':
            visualize.append(visualize)
    print 'picked the Avonex ones'
    visualize = visualize[0:6]
    sys.stdout.flush()
    fig = plt.figure(figsize=(15,4))
    axes = []
    ymax = 0
    for n, index in enumerate(visualize):
        print index
        sys.stdout.flush()
        print np.shape(trainData)
        sys.stdout.flush()
        hist = trainData[index, :]
        print hist
        sys.stdout.flush()
        scan = mri_train[index]
        print 'loading image...'
        sys.stdout.flush()
        t2 = nib.load(scan.images['t2w']).get_data()
        print 'image loaded'
        sys.stdout.flush()
        lesionMaskImg = np.zeros((np.shape(t2)))
            
        for lesion in mri_train[index].lesionList:
            for point in lesion:
                lesionMaskImg[point[0], point[1], point[2]] = 1
                            
        maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)                      
        print 'image masked'
        sys.std.out.flush()
        ax = fig.add_subplot(2, 6, n+1)
        ax.imshow(t2[20:200, 20:200, 30].T, cmap=plt.cm.gray, origin='lower')
        ax.imshow(maskImg[20:200,20:200, 30].T, cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
        ax.axis('off')
        
        print 'making hist'
        sys.stdout.flush()
        axes.append(fig.add_subplot(2, 6, n+7))
        axes[-1].bar(range(np.shape(hist)[0]), hist)
        axes[-1].set_xticks([])
        if np.amax(hist) > ymax:
            ymax = np.amax(hist)
                
        if n == 6:
            for ax2 in axes:
                ax2.set_ylim([0, ymax])
            break
    
    plt.subplots_adjust(wspace=0.01,hspace=0.01)
    plt.savefig('/usr/local/data/adoyle/images/example' + str(example) + '.png')
#    plt.show()
    plt.close()

def removeWorstFeatures(trainData, testData, removeThisRound):

    
    for remove in removeThisRound:
        trainData = np.delete(trainData, remove, 1)
        testData = np.delete(testData, remove, 1)
    
        
    return trainData, testData

def plotScores(scoring, plotTitle):
    try:
        numBars = len(scoring)*4
        bars = []
        ticks = ['TP', 'FP', 'TN', 'FN']
        colours = ['b', 'g', 'r', 'c', 'm', 'y', 'aqua', 'k', 'gold', 'lightgreen']    
        
        fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
    
        for i, (scoreObj, label) in enumerate(scoring):
            x = np.linspace(0, numBars, num=4, dtype='float')
            x = np.add(x, i) #shifts bars over
            y = [np.sum(scoreObj['TP']), np.sum(scoreObj['FP']), np.sum(scoreObj['TN']), np.sum(scoreObj['FN'])]
    #            bars.append(ax2.bar(x, y, color=colours[i], label=label))
            
            print label
            print np.sum(scoreObj['TP']), np.sum(scoreObj['FP']), np.sum(scoreObj['TN']), np.sum(scoreObj['FN'])
            
            print 'sensitivity: ', np.sum(scoreObj['TP']) / (np.sum(scoreObj['TP']) + np.sum(scoreObj['FN']))
            print 'specificity: ', np.sum(scoreObj['TN']) / (np.sum(scoreObj['TN']) + np.sum(scoreObj['FP']))
        
    #        ax2.set_xticks(np.linspace(0, numBars, num=4, endpoint=True) + numBars/8)
    #        ax2.set_xticklabels(ticks)
    ##        ax2.set_title('10-fold average results')
    #        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
        
    
        labels = []
    
    
    #        if "Avonex" in str(foldNum):
    #            ax2.set_title("Drug A results")
    #        elif "Laq" in str(foldNum):
    #            ax2.set_title("Drug B results")
    #        else:
    #            ax2.set_title("Placebo results")
        
        print plotTitle
        
        plots = []
        for i, (scoreObj, label) in enumerate(scoring):
            labels.append(label)
            scatterPoint = ax.scatter(np.sum(scoreObj['TN']) / (np.sum(scoreObj['TN']) + np.sum(scoreObj['FP'])), np.sum(scoreObj['TP']) / (np.sum(scoreObj['TP']) + np.sum(scoreObj['FN'])), marker='x', s=(100,), color=colours[i])
            plots.append(scatterPoint)
            
        ax.set_ylabel('Sensitivity')
        ax.set_xlabel('Specificity')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.title(plotTitle)
        plt.legend(tuple(plots), tuple(labels), loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1, fancybox=True, shadow=True)
        
        plt.savefig('/usr/local/data/adoyle/ss-results-' + str(random.randint(1, 1000)) + '.png', dpi=500)
        plt.show()
    except:
        print "coulnt plot"

def plotGroupDistribution(trainOutcomes, testOutcomes, trainClusterAssignments, testClusterAssignments, n, predictions, method):
    
    #n = len(set(trainClusterAssignments))
    
    trainActive = np.zeros(n)    
    trainInactive = np.zeros(n)    
    
    for (outcome, group) in zip(trainOutcomes, trainClusterAssignments):
        if outcome == 1:
            trainActive[group] += 1
        else:
            trainInactive[group] += 1
    
    testActive = np.zeros(n)    
    testInactive = np.zeros(n)
    
    for (outcome, group) in zip(testOutcomes, testClusterAssignments):
        if outcome == 1:
            testActive[group] += 1
        else:
            testInactive[group] += 1
    
    xticks = np.linspace(1, n, num=n)
    
    bar_width = 0.35    
    
    fig, (ax, ax2, ax3) = plt.subplots(3, figsize=(4,12))
            
    ax.bar(xticks-0.2, trainActive, bar_width, color='r')
    ax.bar(xticks, trainInactive, bar_width, color='g')
    ax.set_xlim([0, n+1])
    ax.set_title('Training - ' + method)
    ax.set_ylabel('Number of Patients')
    
    ax2.bar(xticks-0.2, testActive, bar_width, color='r')
    ax2.bar(xticks, testInactive, bar_width, color='g')
    ax2.set_ylabel('Number of Patients')
    ax2.set_title('Testing - ' + method)
    ax2.set_xlim([0, n+1])
    
    groupTotals = np.zeros(n)
    for i in range(n):
        groupTotals[i] += testActive[i]
        groupTotals[i] += testInactive[i]
    
    print 'group totals:', groupTotals    
    
    groupAccuracy = np.zeros(n)
    for (outcome, prediction, group) in zip(testOutcomes, predictions, testClusterAssignments):
        if outcome == prediction:
            groupAccuracy[group] += 1.0 / groupTotals[group]
    
    ax3.bar(xticks-0.2, groupAccuracy, bar_width)
    ax3.set_xlabel('Group Number')
    ax3.set_ylabel('Accuracy')
    ax3.set_xlim([0, n+1])
    ax3.set_ylim([0, 1])   
    
    
    plt.show()
    plt.close('all')

def examineClusters(groupProbs):
    allProbs = []
    for probs in groupProbs:
        allProbs.append(np.amax(probs))
    
    plt.hist(allProbs)
    plt.show()
    plt.close()
    
def beforeAndAfter():
    plt.close()
    mri_list = context_extraction.loadMRIList()
     
    for i, scan in enumerate(mri_list):
        fig = plt.figure()
        
        subprocess.call(['mnc2nii', scan.newT2, scan.newT2[0:-7] + '.nii'])
        subprocess.call(['gzip', scan.newT2[0:-7] + '.nii'])
        scan.newT2 = scan.newT2[0:-7] + '.nii.gz'
            
        t2 = nib.load(scan.images['t2w'][0:-7] + '.nii.gz').get_data()    
        newT2 = nib.load(scan.newT2).get_data()
            
        lesionPoints = nib.load(scan.lesions).get_data()
        lesionList = list(np.asarray(np.nonzero(lesionPoints)).T)
        
        newLesionPoints = nib.load(scan.futureLabels).get_data()
        newLesionList = list(np.asarray(np.nonzero(newLesionPoints)).T)
        
        lesionImg = np.zeros(np.shape(t2))
        newLesionImg = np.zeros(np.shape(newT2))


        for i, (x, y, z) in enumerate(lesionList):
            lesionImg[z,y,x] = 1
            
            
        for i, (x, y, z) in enumerate(newLesionList):
            newLesionImg[z,y,x] = 1
        
        maskImg = np.ma.masked_where(lesionImg == 0, np.ones(np.shape(lesionImg))*5000)
        newMaskImg = np.ma.masked_where(newLesionImg == 0, np.ones(np.shape(newLesionImg))*5000)
        
    
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(t2[20:200, 20:200, 30].T, cmap=plt.cm.gray, origin='lower')
        ax.imshow(maskImg[20:200, 20:200, 30].T, cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
        ax.axis('off')
            
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(newT2[20:200, 20:200, 30].T, cmap=plt.cm.gray, origin='lower')
        ax.imshow(newMaskImg[20:200, 20:200, 30].T, cmap=plt.cm.autumn, interpolation = 'nearest', alpha=0.4, origin='lower')
        ax.axis('off')  
        
        plt.savefig('/usr/local/data/adoyle/images/before-after-' + str(i) + '.png', dpi=500)
        plt.show()        
        plt.close()        
    
    for i, scan in enumerate(mri_list):
        fig = plt.figure(figsize=(12,4))
        
        img = {}
        for j, mod in enumerate(modalities):
            scan.images[mod] = scan.images[mod][0:-7] + '.nii.gz'
            img[mod] = nib.load(scan.images[mod]).get_data()
            ax = fig.add_subplot(1, 4, j+1)
            ax.imshow(img[mod][20:170, 20:200, 30].T, cmap=plt.cm.gray, origin='lower')
            ax.axis('off')
            ax.set_xlabel(mod)
        
#        plt.tight_layout()
        plt.savefig('/usr/local/data/adoyle/images/allmods-' + scan.uid + '.png', dpi=500)
        plt.show()
        plt.close()
        
    for i, scan in enumerate(mri_list):
        fig = plt.figure(figsize=(12,4))
        
        img = {}
        for j, mod in enumerate(modalities):
            img[mod] = nib.load(scan.rawImages[mod]).get_data()
            ax = fig.add_subplot(1, 4, j+1)
            ax.imshow(img[mod][30, 10:210, 10:180], cmap=plt.cm.gray, origin='lower')
            ax.axis('off')
            ax.set_xlabel(mod)
        
#        plt.tight_layout()
        plt.savefig('/usr/local/data/adoyle/images/rawmods-' + scan.uid + '.png', dpi=500)
        plt.show()
        plt.close()

def pruneFeatures(trainData, testData):

    featureCounts = {}
    for s, size in enumerate(sizes):
#        print np.shape(trainData[size])
        featureCounts[size] = np.zeros((np.shape(trainData[size])[1]))
        
    for s, size in enumerate(sizes):
        testData[size] = testData[size][:, (trainData[size] != 0).sum(axis=0) >= 10]
        trainData[size] = trainData[size][:, (trainData[size] != 0).sum(axis=0) >= 10]
        
#    this is very slow!!
        
#    for s, size in enumerate(sizes):
#        for i in range(np.shape(trainData[size])[1]):
#            featureCounts[size][i] = np.sum(trainData[size][:,i])
#
#    for s, size, in enumerate(sizes):
#        r = range(np.shape(trainData[size])[1])[::-1]
#        for i in r:
#            if featureCounts[size][i] == 0:
#                trainData[size] = np.delete(trainData[size], i, 1)
#                testData[size] = np.delete(testData[size], i, 1)
    
    if plotFeats:
        fig, ax = plt.subplots(1,4, figsize=(14,4))
        for s, size in enumerate(sizes):
            for d in trainData[size]:
                ax[s].plot(np.ndarray.flatten(trainData[size]))
            
            ax[s].set_title(size)
        
        plt.tight_layout()
        plt.show()
        
    
    return trainData, testData

def separatePatientsByTreatment(mri_train, mri_test, trainData, testData, trainCounts, testCounts):
    trainingPatientsByTreatment = defaultdict(list)
    testingPatientsByTreatment = defaultdict(list)
    
    trainingData = {}
    testingData = {}
    trainLesionCounts = {}
    testLesionCounts = {}

    treatmentCountTrains = {}
    treatmentCountTest = {}
    treatmentIndexTrains = {}
    treatmentIndexTest = {}
    for treatment in treatments:
        treatmentCountTrains[treatment] = 0
        treatmentCountTest[treatment] = 0
        treatmentIndexTrains[treatment] = 0
        treatmentIndexTest[treatment] = 0

    for scan in mri_train:
        treatmentCountTrains[scan.treatment] += 1
    for scan in mri_test:
        treatmentCountTest[scan.treatment] += 1
        
    for treatment in treatments:
        trainingData[treatment] = np.zeros((treatmentCountTrains[treatment], np.shape(trainData)[1]))
        testingData[treatment] = np.zeros((treatmentCountTest[treatment], np.shape(testData)[1]))
        trainLesionCounts[treatment] = np.zeros((treatmentCountTrains[treatment], np.shape(trainCounts)[1]))
        testLesionCounts[treatment] = np.zeros((treatmentCountTest[treatment], np.shape(testCounts)[1]))

    for i, scan in enumerate(mri_train):
        trainingPatientsByTreatment[scan.treatment].append(scan)
        trainingData[scan.treatment][treatmentIndexTrains[scan.treatment],:] = trainData[i,:]
        trainLesionCounts[scan.treatment][treatmentIndexTrains[scan.treatment],:] = trainCounts[i,:]
        treatmentIndexTrains[scan.treatment] += 1
    
    for i, scan in enumerate(mri_test):
        testingPatientsByTreatment[scan.treatment].append(scan)
        testingData[scan.treatment][treatmentIndexTest[scan.treatment],:] = testData[i,:]
        testLesionCounts[scan.treatment][treatmentIndexTest[scan.treatment],:] = testCounts[i,:]
        treatmentIndexTest[scan.treatment] += 1
    
    for treatment in treatments:
        print 'training shape:', treatment, np.shape(trainingData[treatment])
        print 'testing shape:', treatment, np.shape(testingData[treatment])
    
    return trainingPatientsByTreatment, testingPatientsByTreatment, trainingData, testingData, trainLesionCounts, testLesionCounts

# we want to show here where the placebo-trained model failed to predict a patient showing activity
# this means that the drug had an effect, because it messed up our pre-trained prediction
def showWhereTreatmentHelped(pretrained_predictions, predictions, train_data, test_data, train_outcomes, test_outcomes, train_mri, test_mri):
    respondersRight = 0
    respondersWrong = 0
    
    responder_prediction = []
    responder_actual = []
    
    responder_certain_actual = []
    responder_certain_prediction = []
    
    for test_index, (pretrained_prediction, prediction, test_outcome) in enumerate(zip(pretrained_predictions, predictions, test_outcomes)):
        
        if pretrained_prediction[1] > 0.5 and test_outcome == 0:            
            responder_actual.append(1)
        else:
            responder_actual.append(0)
            
        if pretrained_prediction[1] > 0.8 and test_outcome == 0:
            responder_certain_actual.append(1)
        else:
            responder_certain_actual.append(0)
        
        print 'values (probs, drug prediction, actual): ', pretrained_prediction[1], prediction, test_outcome
        
        if pretrained_prediction[1] > 0.5 and prediction[1] < 0.5:
            responder_prediction.append(1)
        else:
            responder_prediction.append(0)
            
        if pretrained_prediction[1] > 0.8 and prediction[1] < 0.8:
            responder_certain_prediction.append(1)
        else:
            responder_certain_prediction.append(0)
            
        if pretrained_prediction[1] > 0.5 and prediction[1] < 0.5 and test_outcome == 0:
            
            scan = test_mri[test_index]
    
            # get chi2 representation, find closest patient
#            distances = chi2_kernel(train_data, test_data[test_index])
#            print 'chi2 distance shape:', np.shape(distances)        
        
#            closest_index = np.argmin(distances[:,0])
            
            t2_test = nib.load(scan.images['t2w']).get_data()
            testLesionPoints = nib.load(scan.lesions).get_data()
            testLesionList = list(np.asarray(np.nonzero(testLesionPoints)).T)

            testLesionImg = np.zeros(np.shape(t2_test))

            for (x, y, z) in testLesionList:
                testLesionImg[z,y,x] = 1
            
                       
            maskImg = np.ma.masked_where(testLesionImg == 0, np.ones(np.shape(testLesionImg))*5000)
            
            n=4
            
            fig, axes = plt.subplots(2, n+1, sharey='row', figsize=(10, 4))

            axes[0,0].set_xticks([])
            axes[0,0].set_yticks([])
            axes[0,0].imshow(t2_test[20:180, 20:200, 30].T, cmap=plt.cm.gray, origin='lower')
            axes[0,0].imshow(maskImg[20:180, 20:200, 30].T, cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
            
            if scan.treatment == "Avonex":
                axes[0,0].set_xlabel('Responder (Drug A)')
            else:
                axes[0,0].set_xlabel('Responder (Drug B)')
            axes[0,0].set_xticks([])
            axes[0,0].set_yticks([])
            
            x = np.linspace(1, len(test_data[test_index]), num=len(test_data[test_index]))
            axes[1,0].bar(x, test_data[test_index])
            axes[1,0].set_xlabel('Lesion-Types')
            
            closest_index = getNClosestMahalanobis(test_data[test_index], n, train_data)

            print "Responder:", scan.uid
            for i, closest in enumerate(closest_index):            
                train_scan = train_mri[closest]
                print "closest:", train_scan.uid
    
                t2_train = nib.load(train_scan.images['t2w']).get_data()
                    
                trainLesionPoints = nib.load(train_scan.lesions).get_data()
                trainLesionList = list(np.asarray(np.nonzero(trainLesionPoints)).T)
                trainLesionImg = np.zeros(np.shape(t2_train))

                for (x, y, z) in trainLesionList:
                    trainLesionImg[z,y,x] = 1

                newMaskImg = np.ma.masked_where(trainLesionImg == 0, np.ones(np.shape(trainLesionImg))*5000)
                axes[0,i+1].set_xticks([])
                axes[0,i+1].set_yticks([])
                
                axes[0,i+1].imshow(t2_train[20:180, 20:200, 30].T, cmap=plt.cm.gray, origin='lower')
                axes[0,i+1].imshow(newMaskImg[20:180, 20:200, 30].T, cmap=plt.cm.autumn, interpolation = 'nearest', alpha=0.4, origin='lower')
                axes[0,i+1].set_title('Close Patient')                
                if scan.newT2 > 0:
                    axes[0,i+1].set_xlabel('(active)')
                else:
                    axes[0,i+1].set_xlabel('(non-active)')
                axes[0,i+1].set_xticks([])
                axes[0,i+1].set_yticks([])
                
                
                x = np.linspace(1, len(train_data[closest]), num=len(train_data[closest]))
                
                axes[1,i+1].set_xlabel('Lesion-Types')
                axes[1,i+1].bar(x, train_data[closest])
            
            plt.savefig('/usr/local/data/adoyle/images/responder-' + scan.uid + '.png', dpi=500)
            plt.show()
            plt.close()

            
            respondersRight += 1
        
        if pretrained_prediction[1] > 0.5 and prediction[1] < 0.5 and test_outcome == 1:
            respondersWrong += 1
    
    responder_score = bol_classifiers.calculateScores(responder_prediction, responder_actual)
    responder_uncertain_score = bol_classifiers.calculateScores(responder_prediction, responder_certain_actual)
    responder_certain_score = bol_classifiers.calculateScores(responder_certain_prediction, responder_actual)
    responder_more_certain_score = bol_classifiers.calculateScores(responder_certain_prediction, responder_certain_actual)
    
    print "Responders(right, wrong)", respondersRight, respondersWrong
    return respondersRight, respondersWrong, responder_score, responder_uncertain_score, responder_certain_score, responder_more_certain_score


def justTreatmentGroups():
    start = time.time()
    mri_list = pkl.load(open('/usr/local/data/adoyle/mri_list.pkl', 'rb'))
    mri_list, without_clinical = load_data.loadClinical(mri_list)
        
    outcomes = getOutcomes(mri_list)
    
    kf = StratifiedKFold(outcomes['newT2'], n_folds=50, shuffle=True)
#    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
#    treatments = ['Placebo']

    failedFolds =0

    respondersRight = {}
    respondersWrong = {}

    certainNumber = defaultdict(dict)
    certainCorrect = defaultdict(dict)
    
    certainNumberPre = defaultdict(dict)
    certainCorrectPre = defaultdict(dict)


    scores = defaultdict(dict)

    knnEuclideanScores = defaultdict(dict)
    knnMahalanobisScores = defaultdict(dict)
    subdivisionScores = defaultdict(dict)
    softSubdivisionScores = defaultdict(dict)
    chi2Scores = defaultdict(dict)
    chi2svmScores = defaultdict(dict)
    featureScores = defaultdict(dict)
    svmLinScores = defaultdict(dict)
    svmRadScores = defaultdict(dict)
    rvmScores = defaultdict(dict)
    preTrainedKnnEuclideanScores = defaultdict(dict)
    preTrainedFeatureScores = defaultdict(dict)
    preTrainedSvmLinScores = defaultdict(dict)
    preTrainedSvmRadScores = defaultdict(dict)
    
    countingScores = defaultdict(dict)
    
    bestScores = defaultdict(dict)
    bestKnnEuclideanScores = defaultdict(dict)
    bestKnnMahalanobisScores = defaultdict(dict)
    bestSubdivisionScores = defaultdict(dict)
    bestSoftSubdivisionScores = defaultdict(dict)
    bestChi2Scores = defaultdict(dict)
    bestChi2svmScores = defaultdict(dict)
    bestFeatureScores = defaultdict(dict)
    bestSvmLinScores = defaultdict(dict)
    bestSvmRadScores = defaultdict(dict)
    bestRvmScores = defaultdict(dict)
    bestPreTrainedKnnEuclideanScores = defaultdict(dict) 
    bestPreTrainedFeatureScores = defaultdict(dict)
    bestPreTrainedSvmLinScores = defaultdict(dict)
    bestPreTrainedSvmRadScores = defaultdict(dict)
    
    probScores = defaultdict(dict)
    allProbScores = defaultdict(dict)
    
    responderScores = defaultdict(dict)
    responderHighProbScores = defaultdict(dict)
    countScores = defaultdict(dict)
    
    r1 = defaultdict(dict)
    r2 = defaultdict(dict)
    r3 = defaultdict(dict)
    r4 = defaultdict(dict)
    
    for treatment in treatments:
        scores[treatment] = defaultdict(list)
        knnEuclideanScores[treatment] = defaultdict(list)
        knnMahalanobisScores[treatment] = defaultdict(list)
        subdivisionScores[treatment] = defaultdict(list)
        softSubdivisionScores[treatment] = defaultdict(list)
        chi2Scores[treatment] = defaultdict(list)
        chi2svmScores[treatment] = defaultdict(list)
        featureScores[treatment] = defaultdict(list)
        svmLinScores[treatment] = defaultdict(list)
        svmRadScores[treatment] = defaultdict(list)
        rvmScores[treatment] = defaultdict(list)
        preTrainedKnnEuclideanScores[treatment] = defaultdict(list)
        preTrainedFeatureScores[treatment] = defaultdict(list)
        bestPreTrainedSvmLinScores[treatment] = defaultdict(list)
        bestPreTrainedSvmRadScores[treatment] = defaultdict(list)
        countingScores[treatment] = defaultdict(list)
        
        bestScores[treatment] = defaultdict(list)
        bestKnnEuclideanScores[treatment] = defaultdict(list)
        bestKnnMahalanobisScores[treatment] = defaultdict(list)
        bestSubdivisionScores[treatment] = defaultdict(list)
        bestSoftSubdivisionScores[treatment] = defaultdict(list)
        bestChi2Scores[treatment] = defaultdict(list)
        bestChi2svmScores[treatment] = defaultdict(list)
        bestFeatureScores[treatment] = defaultdict(list)
        bestSvmLinScores[treatment] = defaultdict(list)
        bestSvmRadScores[treatment] = defaultdict(list)
        bestRvmScores[treatment] = defaultdict(list)
        bestPreTrainedKnnEuclideanScores[treatment] = defaultdict(list)
        bestPreTrainedFeatureScores[treatment] = defaultdict(list)
        preTrainedSvmLinScores[treatment] = defaultdict(list)
        preTrainedSvmRadScores[treatment] = defaultdict(list)
        
        probScores[treatment] = defaultdict(list)
        allProbScores[treatment] = defaultdict(list)
        
        responderScores[treatment] = defaultdict(list)
        responderHighProbScores[treatment] = defaultdict(list)
        countScores[treatment] = defaultdict(list)

        certainNumber[treatment] = 0
        certainCorrect[treatment] = 0
        certainNumberPre[treatment] = 0
        certainCorrectPre[treatment] = 0
        
        respondersRight[treatment] = 0
        respondersWrong[treatment] = 0
        
        r1[treatment] = defaultdict(list)
        r2[treatment] = defaultdict(list)
        r3[treatment] = defaultdict(list)
        r4[treatment] = defaultdict(list)

    for foldNum, (train_index, test_index) in enumerate(kf):

        print foldNum, '/', len(kf)
#        for train_index, test_index in sss.split(mri_list, outcomes['newT2']):
#            mri_train, mri_test = train_test_split(mri_list, test_size=0.18, random_state=5)
        
        mri_train = np.asarray(mri_list)[train_index]
        mri_test = np.asarray(mri_list)[test_index]
        
        trainCounts = load_data.loadLesionNumbers(mri_train)
        testCounts = load_data.loadLesionNumbers(mri_test)
        
        print "training:", len(mri_train)
        #incorporate patients with no clinical data
        train_patients = []
        for scan in mri_train:
            train_patients.append(scan)
        for scan in without_clinical:
            train_patients.append(scan)
    
        print 'loading data...'
        startLoad = time.time()
        numLesionsTrain, lesionSizesTrain, lesionCentroids, brainUids = load_data.getLesionSizes(train_patients)
        trainDataVectors, lbpPCA = load_data.loadAllData(train_patients, numLesionsTrain)
        
        numLesionsTest, lesionSizesTest, lesionCentroids, brainUids = load_data.getLesionSizes(mri_test)
        dataVectorsTest, lbpPCA = load_data.loadAllData(mri_test, numLesionsTest, lbpPCA=lbpPCA)
        
        print 'loading data took', (time.time() - startLoad)/60.0, 'minutes'
        
        print 'removing infrequent features...'
        startPruneTime = time.time()
        prunedDataTrain = []
        prunedDataTest = []
        
        for dTrain, dTest in zip(trainDataVectors, dataVectorsTest):
            dTrainPruned, dTestPruned = pruneFeatures(dTrain, dTest)
            prunedDataTrain.append(dTrainPruned)
            prunedDataTest.append(dTestPruned)
        
        del trainDataVectors
        del dataVectorsTest
        print "it took", (time.time() - startPruneTime)/60.0, "minutes"
    
        print 'learning bag of lesions...'

        startBol = time.time()
        allTrainData, clusters, pcas, subtypeShape, brainIndices, lesionIndices = createRepresentationSpace(train_patients, prunedDataTrain, lesionSizesTrain, len(mri_train), lesionCentroids, examineClusters=False)
        elapsedBol = time.time() - startBol
        print str(elapsedBol / 60), 'minutes to learn BoL.'                                
                    
#           tfidfTrans = TfidfTransformer()
#           allTrainData = tfidfTrans.fit_transform(allTrainData).toarray()        
   
#                    pca = None
    #    ica = FastICA()
    #    ica.fit(data)
    #    data = ica.transform(data)

#                    pca = PCA(n_components=120, copy=False)
#                    data = pca.fit_transform(data)
#                    print 'explained variance ratio:', np.sum(pca.explained_variance_ratio_)

        print 'transforming test data to bag of lesions representation...'    
        allTestData = testRepresentationSpace(mri_test, prunedDataTest, lesionSizesTest, clusters, pcas)        
        
#                    allTestData = tfidfTrans.transform(allTestData).toarray()                 
        
#            allTrainData, allTestData, lesionSizeFeatures = pruneFeatures(allTrainData, allTestData)
        
        print 'splitting data up by treatment group'
        trainingPatientsByTreatment, testingPatientsByTreatment, trainingData, testingData, trainCounts, testCounts = separatePatientsByTreatment(mri_train, mri_test, allTrainData, allTestData, trainCounts, testCounts)
        
        featuresToRemove = None
        
        
        c = None
        print 'grouping patients'
        for treatment in treatments:
            try:
                scoreThisFold = True
                
                trainData = trainingData[treatment]
                testData = testingData[treatment]
                
                trainDataCopy = trainData
                testDataCopy = testData
                
                trainOutcomes = getOutcomes(trainingPatientsByTreatment[treatment])
                testOutcomes = getOutcomes(testingPatientsByTreatment[treatment])
    
                remove_worst_features = True
                if remove_worst_features:
                    if treatment == "Placebo":
                        print 'selecting features...'
                        bestTrainData, bestTestData, featuresToRemove = bol_classifiers.randomForestFeatureSelection(trainDataCopy, testDataCopy, trainOutcomes['newT2'], testOutcomes['newT2'], 12)  
                    else:
                        print 'using previously determined best features'
        #                print featuresToRemove
                        bestTrainData, bestTestData = removeWorstFeatures(trainDataCopy, testDataCopy, featuresToRemove)
                else:
                    bestTrainData = trainDataCopy
                    bestTestData  = testDataCopy
    
    
                print 'train, test data shape:', np.shape(bestTrainData), np.shape(bestTestData)
    
    #            trainClusterData, validationData = train_test_split(bestTrainData, test_size=0.1, random_state=5)
    #                        ratio = len(trainOutcomes['newT2'])/ float(np.sum(trainOutcomes['newT2']))
    #                        smote = SMOTE(ratio=ratio, kind='regular', k=3)
    #                    
    #                        print 'oversampling data...'
    #                        trainData, trainOutcomes['newT2'] = smote.fit_transform(trainData, np.asarray(trainOutcomes['newT2']))
                
    #            numClusters = []
    #            bics = []
    #            aics = []
    #            for k in range(2, 12):
    #                clust = GMM(n_components = k, covariance_type = 'full')
    #                clust.fit(trainClusterData)
    #                
    #                numClusters.append(k)
    #                bic = clust.bic(validationData)
    #                aic = clust.aic(validationData)
    #                
    #                bics.append(bic)
    #                aics.append(aic)
    ##                            print k, bic  
    #
    #                nClusters = numClusters[np.argmin(bics)]
    #                
    #            plt.plot(numClusters, bics)
    #            plt.plot(numClusters, aics)
    #            plt.xlabel("Number of Clusters")
    #            plt.ylabel("Information Criterion Value (Lower is better)")
    #            plt.show()
    ##                if nClusters == 2:
    ##                    nClusters = 5
    #            
    #            
    #            c = GMM(n_components = nClusters, covariance_type = 'full')
    #            c.fit(bestTrainData)
    #
    #            trainClusterAssignments = c.predict(bestTrainData)
    #            testClusterAssignments = c.predict(bestTestData)
    #       
    #            trainGroupProbs = c.predict_proba(bestTrainData)
    #            testGroupProbs = c.predict_proba(bestTestData)
    #            
    #            
    #            visualizePatientGroups(trainingPatientsByTreatment[treatment], trainData, trainClusterAssignments, subtypeShape)
    
    
                # comparing classifiers
                if treatment == "Placebo":
                    (bestFeatureScore, bestFeaturePredictions, placebo_rf), (probScore, probPredicted), (correct, total) = bol_classifiers.featureClassifier(bestTrainData, bestTestData, trainOutcomes, testOutcomes, subtypeShape, train_patients, mri_test, brainIndices, lesionIndices, len(mri_list))   
                   
                    (bestChi2Score, bestChi2Predictions), (bestChi2svmscore, bestChi2svmPredictions) = bol_classifiers.chi2Knn(bestTrainData, bestTestData, trainOutcomes, testOutcomes)
                    (bestSvmLinScore, bestSvmLinPredictions, svm1), (bestSvmRadScore, bestSvmRadPredictions, svm2) = bol_classifiers.svmClassifier(bestTrainData, bestTestData, trainOutcomes, testOutcomes)
                    (bestKnnEuclideanScoreVals, bestEuclideanPredictions), (bestKnnMahalanobisScoreVals, bestMahalanobisPredictions) = bol_classifiers.knn(bestTrainData, trainOutcomes, bestTestData, testOutcomes)
                    
                    
                    (featureScore, featurePredictions, meh), (allProbScore, allprobPredicted), (allCorrect, allTotal) = bol_classifiers.featureClassifier(trainData, testData, trainOutcomes, testOutcomes, subtypeShape, train_patients, mri_test, brainIndices, lesionIndices, len(mri_list))   
                    
                    (countingScore, countingPredictions, placebo_nb) = bol_classifiers.countingClassifier(trainCounts[treatment], testCounts[treatment], trainOutcomes, testOutcomes)
                
                # drugged patients
                else:
                    # natural course ms model
                    (bestPreTrainedFeatureScore, bestPreTrainedFeaturePredictions, meh), (pretrainedProbScore, pretrainedProbPredicted), (correct, total) = bol_classifiers.featureClassifier(bestTrainData, bestTestData, trainOutcomes, testOutcomes, subtypeShape, train_patients, mri_test, brainIndices, lesionIndices, len(mri_list), placebo_rf)   
                    
                    #new model on drugged patients
                    (bestFeatureScore, bestFeaturePredictions, meh), (probScore, probDrugPredicted), (correct, total) = bol_classifiers.featureClassifier(bestTrainData, bestTestData, trainOutcomes, testOutcomes, subtypeShape, train_patients, mri_test, brainIndices, lesionIndices, len(mri_list))   
    
    #                (bestPreTrainedKnnEuclideanScoreVals, bestEuclideanPredictions, meh) = bol_classifiers.knn(bestTrainData, trainOutcomes, bestTestData, testOutcomes, clf)
    #                (bestPreTrainedSvmLinScore, bestPreTraindSvmLinearPredictions, meh), (bestPreTrainedSvmRadScore, bestSvmRadialPredictions, meh) = bol_classifiers.svmClassifier(bestTrainData, bestTestData, trainOutcomes, testOutcomes, svm1, svm2)
                    certainNumber[treatment] += total
                    certainCorrect[treatment] += correct
    
                    right, wrong, r1_score, r2_score, r3_score, r4_score = showWhereTreatmentHelped(pretrainedProbPredicted, probDrugPredicted, bestTrainData, bestTestData, trainOutcomes['newT2'], testOutcomes['newT2'], trainingPatientsByTreatment[treatment], testingPatientsByTreatment[treatment])
                    
                    respondersRight[treatment] += right
                    respondersWrong[treatment] += wrong
                    
                    print 'responders right', respondersRight
                    print 'responders wrong', respondersWrong
                    
                    (responderScore, responderProbs), responderHighProbScore, count_score = bol_classifiers.identifyResponders(bestTrainData, bestTestData, trainOutcomes, testOutcomes, trainCounts[treatment], testCounts[treatment], placebo_rf, placebo_nb) 
                    
                certainNumberPre[treatment] += total
                certainCorrectPre[treatment] += correct
                # full feature set
    #            try:
    #                (softSubdivisionScore, softSubdivisionPredictions) = bol_classifiers.softSubdividePredictGroups(trainData, trainClusterAssignments, trainOutcomes, testData, testGroupProbs, testOutcomes, nClusters)
    #            except:
    #                print 'ERROR: Couldnt do this one'
    
    
    #            (chi2Score, chi2Predictions), (chi2svmscore, chi2svmPredictions) = bol_classifiers.chi2Knn(trainData, testData, trainOutcomes, testOutcomes)
    #            (svmLinearScore, svmLinearPredictions), (svmRadialScore, svmRadialPredictions) = bol_classifiers.svmClassifier(trainData, testData, trainOutcomes, testOutcomes)
    #            (bayesScoreVals, bayesPredictions) = bol_classifiers.predictOutcomeGivenGroups(trainGroupProbs, trainOutcomes, testGroupProbs, testOutcomes, testClusterAssignments)        
    #            (knnEuclideanScoreVals, euclideanPredictions), (knnMahalanobisScoreVals, mahalanobisPredictions) = bol_classifiers.knn(trainData, trainOutcomes, testData, testOutcomes)
    #            (knnEuclideanScoreVals, euclideanPredictions) = bol_classifiers.knn(trainData, trainOutcomes, testData, testOutcomes)
    #                try:
    #                    (rvmScoreVals, rvmPredictions) = bol_classifiers.rvmClassifier(trainData, testData, trainOutcomes, testOutcomes)
    #                except:
    #                    pass
                
    #            plotGroupDistribution(trainOutcomes['newT2'], testOutcomes['newT2'], trainClusterAssignments, testClusterAssignments, nClusters, softSubdivisionPredictions, 'Activity Subgroups')
                
                for scoreMet in scoringMetrics + ['sensitivity', 'specificity']:
    ##                scores[treatment][scoreMet].append(bayesScoreVals['newT2'][scoreMet])
    #                knnEuclideanScores[treatment][scoreMet].append(knnEuclideanScoreVals['newT2'][scoreMet])
    #                knnMahalanobisScores[treatment][scoreMet].append(knnMahalanobisScoreVals['newT2'][scoreMet])
    ##                softSubdivisionScores[treatment][scoreMet].append(softSubdivisionScore['newT2'][scoreMet])
    ##                chi2Scores[treatment][scoreMet].append(chi2Score['newT2'][scoreMet])
    ##                chi2svmScores[treatment][scoreMet].append(chi2svmscore['newT2'][scoreMet])
                    featureScores[treatment][scoreMet].append(featureScore['newT2'][scoreMet])
    ##                svmLinScores[treatment][scoreMet].append(svmLinearScore['newT2'][scoreMet])
    ##                svmRadScores[treatment][scoreMet].append(svmRadialScore['newT2'][scoreMet])
    ##                    rvmScores[treatment][scoreMet].append(rvmScoreVals['newT2'][scoreMet])
                    
                    #bad classifiers
                    bestKnnEuclideanScores[treatment][scoreMet].append(bestKnnEuclideanScoreVals['newT2'][scoreMet])
                    bestKnnMahalanobisScores[treatment][scoreMet].append(bestKnnMahalanobisScoreVals['newT2'][scoreMet])
                    bestChi2Scores[treatment][scoreMet].append(bestChi2Score['newT2'][scoreMet])
                    bestChi2svmScores[treatment][scoreMet].append(bestChi2svmscore['newT2'][scoreMet])
                    bestFeatureScores[treatment][scoreMet].append(bestFeatureScore['newT2'][scoreMet])
                    bestSvmLinScores[treatment][scoreMet].append(bestSvmLinScore['newT2'][scoreMet])
                    bestSvmRadScores[treatment][scoreMet].append(bestSvmRadScore['newT2'][scoreMet])
    
    
                    countingScores[treatment][scoreMet].append(countingScore['newT2'][scoreMet])
                    probScores[treatment][scoreMet].append(probScore[scoreMet])
                    allProbScores[treatment][scoreMet].append(probScore[scoreMet])
                
                    
                    if treatment != "Placebo":
                        preTrainedFeatureScores[treatment][scoreMet].append(bestPreTrainedFeatureScore['newT2'][scoreMet])
    #                    preTrainedKnnEuclideanScores[treatment][scoreMet].append(bestPreTrainedKnnEuclideanScoreVals['newT2'][scoreMet])
    #                    preTrainedSvmLinScores[treatment][scoreMet].append(bestPreTrainedSvmLinScore['newT2'][scoreMet])
    #                    preTrainedSvmRadScores[treatment][scoreMet].append(bestPreTrainedSvmRadScore['newT2'][scoreMet])
                        responderScores[treatment][scoreMet].append(responderScore[scoreMet])
                        responderHighProbScores[treatment][scoreMet].append(responderHighProbScore[scoreMet])
                        countScores[treatment][scoreMet].append(count_score[scoreMet])
                        
                        r1[treatment][scoreMet].append(r1_score[scoreMet])
                        r2[treatment][scoreMet].append(r2_score[scoreMet])
                        r3[treatment][scoreMet].append(r3_score[scoreMet])
                        r4[treatment][scoreMet].append(r4_score[scoreMet])
                        
            except:
                failedFolds += 1
                scoreThisFold = False
            
            if scoreThisFold:
                for treatment in treatments:
                    if treatment == "Placebo":
                        bestScoring = []
                        bestScoring.append((bestKnnEuclideanScores[treatment], "NN-Euclidean"))
                        bestScoring.append((bestKnnMahalanobisScores[treatment], "NN-Mahalanobis"))
                        bestScoring.append((bestChi2Scores[treatment], "NN-$\chi^2$"))
                        
                        
                        bestScoring.append((bestSvmLinScores[treatment], "SVM-Linear"))
                        bestScoring.append((bestSvmRadScores[treatment], "SVM-RBF"))
                        bestScoring.append((bestChi2svmScores[treatment], "SVM-$\chi^2$"))
                        
                        bestScoring.append((bestFeatureScores[treatment], "Random Forest"))
                        bestScoring.append((countingScores[treatment], "Naive Bayes (Lesion Counts)"))
                    
                        plotScores(bestScoring, 'Activity Prediction (Untreated)')
                        
                    if treatment == "Placebo":
                        bestScoring = []
                        
                        bestScoring.append((featureScores[treatment], "Random Forest (all lesions)"))
                        bestScoring.append((allProbScores[treatment], "Random Forest (all lesions, certain)"))
                        
                        bestScoring.append((bestFeatureScores[treatment], "Random Forest (best lesions)"))
                        bestScoring.append((probScores[treatment], "Random Forest (best lesions, certain)"))
        #                plotScores(bestScoring, '')
                
                for treatment in treatments:
                    if treatment == "Avonex":
        #                plotScores([(responderScores[treatment], 'Responders'), (responderHighProbScores[treatment], 'Responders (certain)'), (countScores[treatment], 'Responders (lesion counts)')], "Avonex Responder Prediction")
                        plotScores([(r1[treatment], 'Responders'), (r2[treatment], 'Responders (certain GT)'), (r3[treatment], 'Responders (certain prediction)'), (r4[treatment], 'Responders (all certain)')], "Avonex Responder Prediction")
                    elif treatment == "Laquinimod":
        #                plotScores([(responderScores[treatment], 'Responders'), (responderHighProbScores[treatment], 'Responders (certain)'), (countScores[treatment], 'Responders (lesion counts)')], "Laquinimod Responder Prediction")
                        plotScores([(r1[treatment], 'Responders'), (r2[treatment], 'Responders (certain GT)'), (r3[treatment], 'Responders (certain prediction)'), (r4[treatment], 'Responders (all certain)')], "Laquinimod Responder Prediction")
                
                bestScoring = []
                
                for treatment in treatments:
        #            scoring = []
        ##            scoring.append((softSubdivisionScores[treatment], 'Activity Subgroups'))
        ##            scoring.append((scores[treatment], 'Group Membership'))
        #            scoring.append((knnEuclideanScores[treatment], 'Nearest Neighbour: Euclidean'))
        ##            scoring.append((chi2Scores[treatment], 'Nearest Neighbour: $\chi^2$'))
        ##                scoring.append((knnMahalanobisScores[treatment], 'Nearest Neighbour: Mahalanobis'))
        ##            scoring.append((chi2svmScores[treatment], 'SVM: $\chi^2$'))                
        ##            scoring.append((svmLinScores[treatment], 'SVM: Linear'))
        ##            scoring.append((svmRadScores[treatment], 'SVM: RBF'))
        #            scoring.append((featureScores[treatment], 'Random Forest'))
        ##                scoring.append((rvmScores[treatment], 'RVM: RBF'))
        #        
        #            plotScores(scoring, treatment + 'all features fold ' + str(foldNum))
                    
                    
        #            bestScoring.append((bestSoftSubdivisionScores[treatment], 'Activity Subgroups'))
        #            bestScoring.append((bestScores[treatment], 'Group Membership'))
        #            bestScoring.append((bestKnnEuclideanScores[treatment], 'Nearest Neighbour: Euclidean'))
        #            bestScoring.append((bestChi2Scores[treatment], 'Nearest Neighbour: $\chi^2$'))
        #                bestScoring.append((bestKnnMahalanobisScores[treatment], 'Nearest Neighbour: Mahalanobis'))
        #            bestScoring.append((bestChi2svmScores[treatment], 'SVM: $\chi^2$'))                
        #            bestScoring.append((bestSvmLinScores[treatment], 'SVM: Linear'))
        #            bestScoring.append((bestSvmRadScores[treatment], 'SVM: RBF'))
                    
        
                    if treatment == "Placebo":
                        bestScoring.append((bestFeatureScores[treatment], 'Untreated ($\\alpha=0.5$)'))
                        bestScoring.append((probScores[treatment], 'Untreated ($\\alpha=0.8$)'))
        #                bestScoring.append((countingScores[treatment], 'Naive Bayesian (Lesion Counts)'))
        
                    if treatment == "Avonex":
                        bestScoring.append((preTrainedFeatureScores[treatment], 'Untreated Predictor on Drug A'))
                        bestScoring.append((bestFeatureScores[treatment], 'Drug A ($\\alpha=0.5$)'))
                        bestScoring.append((probScores[treatment], 'Drug A ($\\alpha=0.8$)'))
                        
                    if treatment == "Laquinimod":
                        bestScoring.append((preTrainedFeatureScores[treatment], 'Untreated Predictor on Drug B'))
                        bestScoring.append((bestFeatureScores[treatment], 'Drug B ($\\alpha=0.5$)'))
                        bestScoring.append((probScores[treatment], 'Drug B ($\\alpha=0.8$)'))
                    
                plotScores(bestScoring, "Activity Prediction")
            

                
    print "FAILED FOLDS:", failedFolds

    print 'certain correct pretrained', certainCorrectPre
    print 'certain total pretrained', certainNumberPre

    print 'certain correct', certainCorrect
    print 'certain total', certainNumber
    
    end = time.time()
    elapsed = end - start
    print str(elapsed / 60), 'minutes elapsed.'

if __name__ == "__main__":
#    beforeAndAfter()
    plt.ion()
    justTreatmentGroups()
