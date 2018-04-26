import numpy as np
import pickle as pkl

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import StratifiedKFold, train_test_split

from scipy.spatial.distance import euclidean

import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import nibabel as nib

import collections
from collections import defaultdict

import random
import sys

# these are the modules that I wrote
import context_extraction, load_data
import bol_classifiers

import subprocess

import warnings
warnings.filterwarnings("ignore")


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
visualizeAGroup = True

letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)']

treatments = ['Placebo', 'Laquinimod', 'Avonex']

threads = 8

plotFeats = False
usePCA = False

data_dir = '/data1/users/adoyle/MS-LAQ/MS-LAQ-302-STX/'
    


def getNClosest(candidate, n, allLesionFeatures):

    distance = np.zeros((np.shape(allLesionFeatures)[0]))

    for i, lesionFeatures in enumerate(allLesionFeatures):
        distance[i] = euclidean(candidate, lesionFeatures)
    
    nClosest = distance.argsort()[:n+1]

    return nClosest


def getNClosestMahalanobis(candidate, n, allLesionFeatures):
    distances = np.zeros(np.shape(allLesionFeatures)[0])
    variance = np.var(allLesionFeatures, axis=0)

    for i, example in enumerate(allLesionFeatures):
        distances[i] = np.sum(np.divide((candidate - example), variance)**2)
        sys.stdout.flush()

    nClosest = distances.argsort()[:n]

    return nClosest    

def choose_clusters(feature_data, results_dir):

    n_clusters, bics, aics, clust_search, time_taken = [], [], [], [], []

    cluster_range = range(2, 50)
    clust_search.append('')
    clust_search.append('')

    for k in cluster_range:
        print('trying ' + str(k) + ' clusters...')

        clust_search.append(GaussianMixture(n_components=k, covariance_type='full'))

        start_cluster_time = time.time()
        clust_search[k].fit(feature_data)
        end_cluster_time = time.time()

        time_taken.append((end_cluster_time - start_cluster_time) / 60)
        n_clusters.append(k)

        bics.append(clust_search[k].bic(feature_data))
        aics.append(clust_search[k].aic(feature_data))

        print('it took ' + str(time_taken[-1]) + ' minutes')

    n_lesion_types = n_clusters[np.argmin(bics)]
    print(n_lesion_types, 'is the optimal number of lesion-types!')
    print('total time taken for clustering:', str(np.sum(time_taken)))

    fig, (ax) = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(n_clusters, bics, lw=2, label='Bayesian')
    ax.plot(n_clusters, aics, lw=2, label='Akaike')

    ax.set_xlabel("Lesion-types in model", fontsize=24)
    ax.set_ylabel("Information Criterion", fontsize=24)

    radius = np.var(np.asarray(bics, dtype='float32'))*4

    circle = plt.Circle((n_lesion_types, bics[n_lesion_types]), radius, color='k', lw=4, fill=False)
    ax.add_artist(circle)

    ax.legend(shadow=True, fancybox=True, fontsize=16)
    plt.tight_layout()
    plt.savefig(results_dir + 'choosing_clusters.png', bbox_inches='tight')
    plt.close()

    return n_lesion_types


def learn_bol(mri_list, feature_data, n_lesion_types, numWithClinical, results_dir, fold_num):
    type_examples = []

    c = GaussianMixture(n_components=n_lesion_types, covariance_type='full')
    c.fit(feature_data)

    cluster_assignments = c.predict(feature_data)
    cluster_probabilities = c.predict_proba(feature_data)

    bol_representation = np.zeros((len(mri_list), n_lesion_types), dtype='float32')

    # maintain a list of indices for each cluster in each size
    for n in range(n_lesion_types):
        type_examples.append([])

    lesion_idx = 0
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
            feature_values = feature_data[lesion_idx, ...]
            lesion_type_distribution = cluster_probabilities[lesion_idx, ...]

            bol_representation[i, :] += lesion_type_distribution

            lesion_type = np.argmax(lesion_type_distribution)

            type_examples[lesion_type].append((scan, j, feature_values, lesion_type_distribution))

            lesion_idx += 1

    for lesion_type_idx in range(n_lesion_types):
        print('Number of lesions in type', lesion_type_idx, ':', len(type_examples[lesion_type_idx]))

    print('Lesion-type probabilities shape:', cluster_probabilities.shape)

    if fold_num%1 == 0:
        n = 6
        for k in range(n_lesion_types):
            if len(type_examples[k]) > n:
                plt.figure(1, figsize=(15, 15))

                random.shuffle(type_examples[k])

                for i, example in enumerate(type_examples[k][0:n]):
                    scan, lesion_index, feature_val, cluster_probs = example[0], example[1], example[2], example[3]

                    img = nib.load(scan.images['t2w']).get_data()
                    lesionMaskImg = np.zeros((np.shape(img)))

                    for point in scan.lesionList[lesion_index]:
                        lesionMaskImg[point[0], point[1], point[2]] = 1

                    x, y, z = [int(np.mean(xxx)) for xxx in zip(*scan.lesionList[lesion_index])]

                    maskImg = np.ma.masked_where(lesionMaskImg == 0,
                                                 np.ones((np.shape(lesionMaskImg))) * 5000)
                    maskSquare = np.zeros((np.shape(img)))
                    maskSquare[x, y-20:y+20, z-20] = 1
                    maskSquare[x, y-20:y+20, z+20] = 1
                    maskSquare[x, y-20, z-20:z+20] = 1
                    maskSquare[x, y+20, z-20:z+20] = 1

                    square = np.ma.masked_where(maskSquare == 0, np.ones(np.shape(maskSquare)) * 5000)

                    lesionMaskPatch = maskImg[x, y-20:y+20, z-20:z+20]
                    ax = plt.subplot(4, n, i + 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.imshow(img[x, 20:200, 20:175], cmap=plt.cm.gray, interpolation='nearest', origin='lower')
                    ax.imshow(maskImg[x, 20:200, 20:175], cmap=plt.cm.autumn, interpolation='nearest', alpha=0.25, origin='lower')
                    ax.imshow(square[x, 20:200, 20:175], cmap=plt.cm.autumn, interpolation='nearest', origin='lower')

                    ax2 = plt.subplot(4, n, i + 1 + n)
                    ax2.imshow(img[x, y-20:y+20, z-20:z+20], cmap=plt.cm.gray, interpolation='nearest', origin='lower')
                    ax2.imshow(lesionMaskPatch, cmap=plt.cm.autumn, alpha=0.25, interpolation='nearest', origin='lower')
                    # ax2.axes.get_yaxis().set_visible(False)
                    ax2.set_yticks([])
                    ax2.set_xticks([])
                    ax2.set_xlabel(letters[i])

                    # x = np.linspace(1, feature_data.shape[1], num=feature_data.shape[1])
                    ax3 = plt.subplot(4, n, i + 1 + 2*n)
                    # ax3.bar(x, feature_val, color='darkred')

                    x_context = np.arange(44)
                    x_rift = np.arange(44, 60)
                    x_lbp = np.arange(60, 96)
                    x_intensity = np.arange(96, 104)
                    x_size = 104

                    ticks = [x_context[-1] / 2, ((x_rift[-1] - x_rift[0]) / 2) + x_rift[0], ((x_lbp[-1] - x_lbp[0]) / 2) + x_lbp[0], ((x_intensity[-1] - x_intensity[0]) / 2) + x_intensity[0], x_size]
                    tick_labels = ['C', 'RIFT', 'LBP', 'I', 'S']

                    ax3.bar(x_context, feature_val[x_context], color='r')
                    ax3.bar(x_rift, feature_val[x_rift], color='g')
                    ax3.bar(x_lbp, feature_val[x_lbp], color='b')
                    ax3.bar(x_intensity, feature_val[x_intensity], color='orange')
                    ax3.bar(x_size, feature_val[x_size], color='m')

                    ax3.set_xticks(ticks)
                    ax3.set_xticklabels(tick_labels)

                    # data = {}
                    # data['context'] = {}
                    # data['context']['ICBM Prior'] = feature_val[0:4]
                    # data['context']['Lesion Prior'] = feature_val[4]
                    # data['context']['Catani Prior'] = feature_val[5:44]
                    #
                    # data['RIFT'] = {}
                    # data['RIFT']['T1w'] = feature_val[44:48]
                    # data['RIFT']['T2w'] = feature_val[48:52]
                    # data['RIFT']['PDw'] = feature_val[52:56]
                    # data['RIFT']['FLR'] = feature_val[56:60]
                    #
                    # data['LBP'] = {}
                    # data['LBP']['T1w'] = feature_val[60:69]
                    # data['LBP']['T2w'] = feature_val[69:78]
                    # data['LBP']['PDw'] = feature_val[78:87]
                    # data['LBP']['FLR'] = feature_val[87:96]
                    #
                    # data['intensity'] = {}
                    # data['intensity']['T1w'] = feature_val[96:98]
                    # data['intensity']['T2w'] = feature_val[98:100]
                    # data['intensity']['PDw'] = feature_val[100:102]
                    # data['intensity']['FLR'] = feature_val[102:104]
                    #
                    # data['size'] = {}
                    # data['size']['vox'] = feature_val[104]
                    #
                    # label_group_bar(ax3, data)
                    ax3.set_ylim([0, 1])
                    ax3.set_yticks([])

                    # for tick in ax3.get_xticklabels():
                    #     tick.set_rotation(45)

                    y = np.linspace(1, cluster_probabilities.shape[1], num=cluster_probabilities.shape[1])
                    ax4 = plt.subplot(4, n, i + 1 + 3*n)
                    ax4.bar(y, cluster_probs, color='darkorange')
                    ax4.set_ylim([0, 1])
                    ax4.set_yticks([])


                    if i == 0:
                        ax.set_ylabel('Lesion', fontsize=24)
                        ax2.set_ylabel('Close-up', fontsize=24)
                        ax3.set_ylabel('Feature values', fontsize=24)
                        ax4.set_ylabel('Lesion-type prob.', fontsize=24)

                plt.subplots_adjust(wspace=0.01)
                plt.savefig(results_dir + 'fold_' + str(fold_num) + '_lesion_type_' + str(k) + '.png', dpi=600, bbox_inches='tight')
                plt.clf()

    if fold_num % 10 == 0:
        try:
            fig, (ax) = plt.subplots(1, 1, figsize=(6, 4))

            bins = np.linspace(0, n_lesion_types, num=n_lesion_types+1)
            histo = np.histogram(cluster_assignments, bins=bins)

            # print('bins', bins)
            # print('histo', histo[0])
            ax.bar(bins[:-1], histo[0])

            plt.tight_layout()
            plt.savefig(results_dir + 'lesion-types-hist_fold_' + str(fold_num) + '.png', bbox_inches='tight')
        except Exception as e:
            print(e)
            print('Error generating lesion-type histogram for this fold')

    return bol_representation[0:numWithClinical, :], c


def project_to_bol(mri_list, feature_data, c):
    lesion_types = c.predict_proba(feature_data)

    bol_representation = np.zeros((len(mri_list), lesion_types.shape[-1]))

    lesion_idx = 0
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
            bol_representation[i, :] += lesion_types[lesion_idx, :]
            lesion_idx += 1

    return bol_representation


def createRepresentationSpace(mri_list, dataVectors, lesionSizes, numWithClinical, lesionCentroids, examineClusters=False):
    subtypeShape, clusters, lesionTypes = [], [], []
    brainIndices, lesionIndices, brainsOfType, lesionsOfType = {}, {}, {}, {}
    
    for m, size in enumerate(sizes):
        brainIndices[size], lesionIndices[size], brainsOfType[size], lesionsOfType[size] = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

    for m, size in enumerate(sizes):
        subtypeShape.append( () )
        subtypeShape[m] += (len(mri_list),)
        
        clusterAssignments = []
        clusterProbabilities = []
        clusters.append([])

        for d, data in enumerate(dataVectors):
            lesionFeatures = data[size]
            print("START OF", sizes[m], feats[d])
            print('lesion feature shape:', np.shape(lesionFeatures))
   
            numClusters, bics, aics, scores, clustSearch = [], [], [], [], []
            clustSearch.append("")
            clustSearch.append("")
            
            clusterData, validationData = train_test_split(lesionFeatures, test_size=0.3, random_state=5)
            for k in range(2,4):
                print('trying ' + str(k) + ' clusters...')
                clustSearch.append(GaussianMixture(n_components = k, covariance_type = 'full'))
                clustSearch[k].fit(clusterData)
                            
                numClusters.append(k)
                bics.append(clustSearch[k].bic(validationData))
                aics.append(clustSearch[k].aic(validationData))
                scores.append(np.mean(clustSearch[k].score(validationData)))

                
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
            
            print("Selected " + str(nClusters) + " clusters for " + feats[d] + " in " + sizes[m] + " lesions")
            sys.stdout.flush()
            
            c = GaussianMixture(n_components = nClusters, covariance_type = 'full')
            c.fit(lesionFeatures)
            
            subtypeShape[m] += (nClusters, )
        
            clusterAssignments.append(c.predict(lesionFeatures))
            clusterProbabilities.append(c.predict_proba(lesionFeatures))
            clusters[m].append(c)

        lesionTypes.append(np.zeros(subtypeShape[m]))
        
#        randomLesionType = (np.random.randint(shape[1]), np.random.randint(shape[2]), np.random.randint(shape[3]), np.random.randint(shape[4]), m)

        print("Subtypes for " + sizes[m] + ": ", subtypeShape[m])
        print("Combining lesion subtypes...")

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
                                img = nib.load(scan.images['t2w']).get_data()
                                lesionMaskImg = np.zeros((np.shape(img)))
                                
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

                            plt.subplots_adjust(wspace=0.01,hspace=0.01)
                            plt.savefig(data_dir + 'images/t2lesions-'+ size + '-' + ''.join((str(f1),str(f2),str(f3),str(f4))) + '.png', dpi=600)

    pcas, lesionFlat = [], []

    if usePCA:
        print("applying PCA...")
        
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
        clusterProbabilities = []
        
        subtypeShape.append( () )
        
        subtypeShape[m] += (len(mri_list),)
        subtypeShape[m] += tuple(c.n_components for c in clusters[m])
        
        lesionTypes.append(np.zeros(subtypeShape[m]))

        for d, data in enumerate(dataVectors):
            lesionFeatures = data[size]
            c = clusters[m][d]
            
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

    print(groups)
    print(xticks)
    print(y)

    ax.bar(xticks, y, align='center')
    ax.set_xticks([])

    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True)

    scale = 1. / ly
    for pos in range(ly + 1):
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
        
        print(np.shape(np.asarray(patientsInGroup).T))
        axs[group].boxplot(np.asarray(patientsInGroup), 1, '')
        axs[group].set_title('Group ' + str(group) + ' histogram')
        axs[group].set_xticks([])
        axs[group].set_yticks([])
        axs[group].set_xlabel('Lesion Type')
        
    plt.savefig(data_dir + 'images/groupHistograms.png', dpi=200)
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


def pruneFeatures(trainData, testData):
    featureCounts = {}
    for s, size in enumerate(sizes):
        #        print np.shape(trainData[size])
        featureCounts[size] = np.zeros((np.shape(trainData[size])[1]))

    for s, size in enumerate(sizes):
        testData[size] = testData[size][:, (trainData[size] != 0).sum(axis=0) >= 10]
        trainData[size] = trainData[size][:, (trainData[size] != 0).sum(axis=0) >= 10]

    if plotFeats:
        fig, ax = plt.subplots(1, 4, figsize=(14, 4))
        for s, size in enumerate(sizes):
            for d in trainData[size]:
                ax[s].plot(np.ndarray.flatten(trainData[size]))

            ax[s].set_title(size)

        plt.tight_layout()
        plt.show()

    return trainData, testData


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
                
                ax2 = fig.add_subplot(2, 6, n+6)
                ax2.bar(range(len(hist)), hist, width=1)
                ax2.set_xticks([0, lesionSizeFeatures['T']/2, lesionSizeFeatures['T'], lesionSizeFeatures['T']+(lesionSizeFeatures['S']/2), lesionSizeFeatures['T']+lesionSizeFeatures['S'], lesionSizeFeatures['T']+lesionSizeFeatures['S']+lesionSizeFeatures['M']/2, lesionSizeFeatures['T']+lesionSizeFeatures['S']+lesionSizeFeatures['M'], lesionSizeFeatures['T']+lesionSizeFeatures['S']+lesionSizeFeatures['M']+lesionSizeFeatures['L']/2, lesionSizeFeatures['T']+lesionSizeFeatures['S']+lesionSizeFeatures['M']+lesionSizeFeatures['L']])
                ax2.set_xticklabels(['', 'T', '', 'S', '', 'M', '', 'L', ''])

                if np.amax(hist) > ymax:
                    ymax = np.amax(hist)
                
            if n == 6:
                for ax2 in axes:
                    ax2.set_ylim([0, ymax])
                break
            
        groupNum = random.randint(0,100)
        print(groupNum)
        plt.subplots_adjust(wspace=0.01,hspace=0.01)
        plt.savefig(data_dir + '/images/patient-groups' + str(groupNum) + '-' + str(g) + '.png', dpi=500)
#        plt.show()
        plt.close()
        
def visualizeWhereTreatmentInfoHelps(example, mri_test, testData, mri_train, trainData):
    print('example', example)
    print('test data', testData[example, :])
    sys.stdout.flush()
    closeOnes = getNClosestMahalanobis(testData[example, ...], 20, trainData)
    print('found closest ones:', closeOnes)
    sys.stdout.flush()
    visualize = []
    for index in closeOnes:
        if mri_train[index].treatment == 'Avonex':
            visualize.append(visualize)
    print('picked the Avonex ones')
    visualize = visualize[0:6]
    sys.stdout.flush()
    fig = plt.figure(figsize=(15,4))
    axes = []
    ymax = 0
    for n, index in enumerate(visualize):
        print(index)
        sys.stdout.flush()
        print(np.shape(trainData))
        sys.stdout.flush()
        hist = trainData[index, :]
        print(hist)
        sys.stdout.flush()
        scan = mri_train[index]
        print('loading image...')
        sys.stdout.flush()
        t2 = nib.load(scan.images['t2w']).get_data()
        print('image loaded')
        sys.stdout.flush()
        lesionMaskImg = np.zeros((np.shape(t2)))
            
        for lesion in mri_train[index].lesionList:
            for point in lesion:
                lesionMaskImg[point[0], point[1], point[2]] = 1
                            
        maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)                      
        print('image masked')
        sys.std.out.flush()
        ax = fig.add_subplot(2, 6, n+1)
        ax.imshow(t2[20:200, 20:200, 30].T, cmap=plt.cm.gray, origin='lower')
        ax.imshow(maskImg[20:200,20:200, 30].T, cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
        ax.axis('off')
        
        print('making hist')
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
    plt.savefig(data_dir + 'images/example' + str(example) + '.png')
#    plt.show()
    plt.close()


def removeWorstFeatures(trainData, testData, removeThisRound):
    for remove in removeThisRound:
        trainData = np.delete(trainData, remove, 1)
        testData = np.delete(testData, remove, 1)

    return trainData, testData


def plotScores(scoring, plotTitle, results_dir):
    try:
        numBars = len(scoring)*4
        colours = ['b', 'g', 'r', 'c', 'm', 'y', 'aqua', 'k', 'gold', 'lightgreen']    
        
        fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    
        for i, (scoreObj, label) in enumerate(scoring):
            x = np.linspace(0, numBars, num=4, dtype='float')
            x = np.add(x, i) #shifts bars over
            y = [np.sum(scoreObj['TP']), np.sum(scoreObj['FP']), np.sum(scoreObj['TN']), np.sum(scoreObj['FN'])]

            print(label)
            print(np.sum(scoreObj['TP']), np.sum(scoreObj['FP']), np.sum(scoreObj['TN']), np.sum(scoreObj['FN']))
            
            print('sensitivity: ', np.sum(scoreObj['TP']) / (np.sum(scoreObj['TP']) + np.sum(scoreObj['FN'])))
            print('specificity: ', np.sum(scoreObj['TN']) / (np.sum(scoreObj['TN']) + np.sum(scoreObj['FP'])))
    
        labels = []
        
        print(plotTitle)
        
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

        plt.savefig(results_dir + '/ss-results-' + str(random.randint(1, 1000)) + '.png', dpi=500, bbox_inches='tight')
    except:
        print("couldnt plot")

    
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
            lesionImg[x,y,z] = 1
            
        for i, (x, y, z) in enumerate(newLesionList):
            newLesionImg[x,y,z] = 1
        
        maskImg = np.ma.masked_where(lesionImg == 0, np.ones(np.shape(lesionImg))*5000)
        newMaskImg = np.ma.masked_where(newLesionImg == 0, np.ones(np.shape(newLesionImg))*5000)

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(t2[:, :, 30], cmap=plt.cm.gray, origin='lower')
        ax.imshow(maskImg[:, :, 30], cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
        ax.axis('off')
            
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(newT2[:, :, 30], cmap=plt.cm.gray, origin='lower')
        ax.imshow(newMaskImg[:, :, 30], cmap=plt.cm.autumn, interpolation = 'nearest', alpha=0.4, origin='lower')
        ax.axis('off')  
        
        plt.savefig(data_dir + 'images/before-after-' + str(i) + '.png', dpi=500)
        plt.show()        
        plt.close()        
    
    for i, scan in enumerate(mri_list):
        fig = plt.figure(figsize=(12,4))
        
        img = {}
        for j, mod in enumerate(modalities):
            scan.images[mod] = scan.images[mod][0:-7] + '.nii.gz'
            img[mod] = nib.load(scan.images[mod]).get_data()
            ax = fig.add_subplot(1, 4, j+1)
            ax.imshow(img[mod][:, :, 30].T, cmap=plt.cm.gray, origin='lower')
            ax.axis('off')
            ax.set_xlabel(mod)
        
#        plt.tight_layout()
        plt.savefig(data_dir + '/allmods-' + scan.uid + '.png', dpi=500)
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
        plt.savefig(data_dir + '/images/rawmods-' + scan.uid + '.png', dpi=500)
        plt.show()
        plt.close()


def separatePatientsByTreatment(mri_train, mri_test, trainData, testData):
    trainingPatientsByTreatment, testingPatientsByTreatment = defaultdict(list), defaultdict(list)
    
    trainingData, testingData = {}, {}

    treatmentCountTrains, treatmentCountTest = {}, {}
    treatmentIndexTrains, treatmentIndexTest = {}, {}

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
        trainingData[treatment] = np.zeros((treatmentCountTrains[treatment], np.shape(trainData)[-1]))
        testingData[treatment] = np.zeros((treatmentCountTest[treatment], np.shape(testData)[-1]))
        # trainLesionCounts[treatment] = np.zeros((treatmentCountTrains[treatment], np.shape(trainCounts)[1]))
        # testLesionCounts[treatment] = np.zeros((treatmentCountTest[treatment], np.shape(testCounts)[1]))

    for i, scan in enumerate(mri_train):
        trainingPatientsByTreatment[scan.treatment].append(scan)
        trainingData[scan.treatment][treatmentIndexTrains[scan.treatment],:] = trainData[i,:]
        # trainLesionCounts[scan.treatment][treatmentIndexTrains[scan.treatment],:] = trainCounts[i,:]
        treatmentIndexTrains[scan.treatment] += 1
    
    for i, scan in enumerate(mri_test):
        testingPatientsByTreatment[scan.treatment].append(scan)
        testingData[scan.treatment][treatmentIndexTest[scan.treatment],:] = testData[i,:]
        # testLesionCounts[scan.treatment][treatmentIndexTest[scan.treatment],:] = testCounts[i,:]
        treatmentIndexTest[scan.treatment] += 1
    
    for treatment in treatments:
        print('training shape:', treatment, np.shape(trainingData[treatment]))
        print('testing shape:', treatment, np.shape(testingData[treatment]))
    
    return trainingPatientsByTreatment, testingPatientsByTreatment, trainingData, testingData

# we want to show here where the placebo-trained model failed to predict a patient showing activity
# this means that the drug had an effect, because it messed up our pre-trained prediction
def showWhereTreatmentHelped(pretrained_predictions, predictions, train_data, test_data, train_outcomes, test_outcomes, train_mri, test_mri, results_dir):
    respondersRight, respondersWrong = 0, 0
    
    responder_prediction, responder_actual, responder_certain_actual, responder_certain_prediction = [], [], [], []

    all_responders_info = []

    for test_index, (pretrained_prediction, prediction, test_outcome) in enumerate(zip(pretrained_predictions, predictions, test_outcomes)):
        
        if pretrained_prediction[1] > 0.5 and test_outcome == 0:            
            responder_actual.append(1)
        else:
            responder_actual.append(0)
            
        if pretrained_prediction[1] > 0.8 and test_outcome == 0:
            responder_certain_actual.append(1)
        else:
            responder_certain_actual.append(0)
        
        print('values (probs, drug prediction, actual): ', pretrained_prediction[1], prediction, test_outcome)
        
        if pretrained_prediction[1] > 0.5 and prediction[1] < 0.5:
            responder_prediction.append(1)
        else:
            responder_prediction.append(0)
            
        if pretrained_prediction[1] > 0.8 and prediction[1] < 0.8:
            responder_certain_prediction.append(1)
        else:
            responder_certain_prediction.append(0)
            
        if pretrained_prediction[1] > 0.8 and prediction[1] < 0.8 and test_outcome == 0:
            scan = test_mri[test_index]
            t2_test = nib.load(scan.images['t2w']).get_data()
            testLesionPoints = nib.load(scan.lesions).get_data()
            testLesionList = list(np.asarray(np.nonzero(testLesionPoints)).T)

            responder_info = dict()
            responder_info['uid'] = scan.uid
            responder_info['treatment'] = scan.treatment
            responder_info['t2_lesions'] = len(testLesionList)
            responder_info['P(A=1|BoL, untr)'] = pretrained_prediction[1]
            responder_info['P(A=0|BoL, tr)'] = prediction[0]
            all_responders_info.append(responder_info)

            testLesionImg = np.zeros(np.shape(t2_test))

            for (x, y, z) in testLesionList:
                testLesionImg[x,y,z] = 1

            maskImg = np.ma.masked_where(testLesionImg == 0, np.ones(np.shape(testLesionImg))*5000)
            n=4
            
            fig, axes = plt.subplots(2, n+1, sharey='row', figsize=(10, 4))
            axes[0,0].set_xticks([])
            axes[0,0].set_yticks([])
            axes[0,0].imshow(t2_test[30, :, :], cmap=plt.cm.gray, origin='lower')
            axes[0,0].imshow(maskImg[30, :, :], cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
            
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

            print("Responder:", scan.uid)
            for i, closest in enumerate(closest_index):            
                train_scan = train_mri[closest]
                print("closest:", train_scan.uid)
    
                t2_train = nib.load(train_scan.images['t2w']).get_data()
                    
                trainLesionPoints = nib.load(train_scan.lesions).get_data()
                trainLesionList = list(np.asarray(np.nonzero(trainLesionPoints)).T)
                trainLesionImg = np.zeros(np.shape(t2_train))

                for (x, y, z) in trainLesionList:
                    trainLesionImg[x,y,z] = 1

                newMaskImg = np.ma.masked_where(trainLesionImg == 0, np.ones(np.shape(trainLesionImg))*5000)
                axes[0,i+1].set_xticks([])
                axes[0,i+1].set_yticks([])
                
                axes[0,i+1].imshow(t2_train[30, :, :], cmap=plt.cm.gray, origin='lower')
                axes[0,i+1].imshow(newMaskImg[30, :, :], cmap=plt.cm.autumn, interpolation = 'nearest', alpha=0.4, origin='lower')
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
            
            plt.savefig(results_dir + 'responder-' + scan.uid + '.png', dpi=500)
            plt.close()
            
            respondersRight += 1
        
        if pretrained_prediction[1] > 0.5 and prediction[1] < 0.5 and test_outcome == 1:
            respondersWrong += 1
    
    responder_score = bol_classifiers.calculateScores(responder_prediction, responder_actual)
    responder_uncertain_score = bol_classifiers.calculateScores(responder_prediction, responder_certain_actual)
    responder_certain_score = bol_classifiers.calculateScores(responder_certain_prediction, responder_actual)
    responder_more_certain_score = bol_classifiers.calculateScores(responder_certain_prediction, responder_certain_actual)
    
    print("Responders(right, wrong)", respondersRight, respondersWrong)

    return respondersRight, respondersWrong, responder_score, responder_uncertain_score, responder_certain_score, responder_more_certain_score, all_responders_info


def justTreatmentGroups():
    start = time.time()
    mri_list = pkl.load(open(data_dir + 'mri_list.pkl', 'rb'))
    mri_list, without_clinical = load_data.loadClinical(mri_list)
        
    outcomes = getOutcomes(mri_list)
    
    kf = StratifiedKFold(outcomes['newT2'], n_folds=50, shuffle=True)
    failedFolds = 0

    respondersRight, respondersWrong = {}, {}

    certainNumber, certainCorrect, certainNumberPre, certainCorrectPre = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)

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
        
        probScores[treatment], allProbScores[treatment] = defaultdict(list), defaultdict(list)
        
        responderScores[treatment], responderHighProbScores[treatment], countScores[treatment] = defaultdict(list), defaultdict(list), defaultdict(list)

        certainNumber[treatment], certainCorrect[treatment], certainNumberPre[treatment], certainCorrectPre[treatment] = 0, 0, 0, 0
        respondersRight[treatment], respondersWrong[treatment] = 0, 0

        r1[treatment], r2[treatment], r3[treatment], r4[treatment] = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

    for foldNum, (train_index, test_index) in enumerate(kf.split(range(len(mri_list)))):
        print(foldNum, '/', len(kf))
        
        mri_train = np.asarray(mri_list)[train_index]
        mri_test = np.asarray(mri_list)[test_index]
        
        trainCounts = load_data.loadLesionNumbers(mri_train)
        testCounts = load_data.loadLesionNumbers(mri_test)
        
        print("training:", len(mri_train))
        #incorporate patients with no clinical data
        train_patients = []
        for scan in mri_train:
            train_patients.append(scan)
        for scan in without_clinical:
            train_patients.append(scan)
    
        print('loading data...')
        startLoad = time.time()
        numLesionsTrain, lesionSizesTrain, lesionCentroids, brainUids = load_data.getLesionSizes(train_patients)
        trainDataVectors, lbpPCA = load_data.loadAllData(train_patients, numLesionsTrain)
        
        numLesionsTest, lesionSizesTest, lesionCentroids, brainUids = load_data.getLesionSizes(mri_test)
        dataVectorsTest, lbpPCA = load_data.loadAllData(mri_test, numLesionsTest, lbpPCA=lbpPCA)
        
        print('loading data took', (time.time() - startLoad)/60.0, 'minutes')
        print('removing infrequent features...')
        startPruneTime = time.time()
        prunedDataTrain = []
        prunedDataTest = []
        
        for dTrain, dTest in zip(trainDataVectors, dataVectorsTest):
            dTrainPruned, dTestPruned = pruneFeatures(dTrain, dTest)
            prunedDataTrain.append(dTrainPruned)
            prunedDataTest.append(dTestPruned)
        
        del trainDataVectors
        del dataVectorsTest
        print("it took", (time.time() - startPruneTime)/60.0, "minutes")
        print('learning bag of lesions...')

        startBol = time.time()
        allTrainData, clusters, pcas, subtypeShape, brainIndices, lesionIndices = createRepresentationSpace(train_patients, prunedDataTrain, lesionSizesTrain, len(mri_train), lesionCentroids, examineClusters=False)
        elapsedBol = time.time() - startBol
        print(str(elapsedBol / 60), 'minutes to learn BoL.')
                    
#       tfidfTrans = TfidfTransformer()
#       allTrainData = tfidfTrans.fit_transform(allTrainData).toarray()
   
#       pca = None
#       ica = FastICA()
#       ica.fit(data)
#       data = ica.transform(data)

#       pca = PCA(n_components=120, copy=False)
#       data = pca.fit_transform(data)
#       print 'explained variance ratio:', np.sum(pca.explained_variance_ratio_)

        print('transforming test data to bag of lesions representation...')
        allTestData = testRepresentationSpace(mri_test, prunedDataTest, lesionSizesTest, clusters, pcas)        
        
#       allTestData = tfidfTrans.transform(allTestData).toarray()
#       allTrainData, allTestData, lesionSizeFeatures = pruneFeatures(allTrainData, allTestData)
        
        print('splitting data up by treatment group')
        trainingPatientsByTreatment, testingPatientsByTreatment, trainingData, testingData, trainCounts, testCounts = separatePatientsByTreatment(mri_train, mri_test, allTrainData, allTestData, trainCounts, testCounts)
        
        featuresToRemove, c = None, None

        print('grouping patients')
        for treatment in treatments:
            try:
                scoreThisFold = True
                
                trainData, testData = trainingData[treatment], testingData[treatment]
                trainDataCopy, testDataCopy = trainData, testData
                trainOutcomes, testOutcomes = getOutcomes(trainingPatientsByTreatment[treatment]), getOutcomes(testingPatientsByTreatment[treatment])

                remove_worst_features = True
                if remove_worst_features:
                    if treatment == "Placebo":
                        print('selecting features...')
                        bestTrainData, bestTestData, featuresToRemove = bol_classifiers.randomForestFeatureSelection(trainDataCopy, testDataCopy, trainOutcomes['newT2'], testOutcomes['newT2'], 12)  
                    else:
                        bestTrainData, bestTestData = removeWorstFeatures(trainDataCopy, testDataCopy, featuresToRemove)
                else:
                    bestTrainData = trainDataCopy
                    bestTestData  = testDataCopy

                print('train, test data shape:', np.shape(bestTrainData), np.shape(bestTestData))
    
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
                    
                    print('responders right', respondersRight)
                    print('responders wrong', respondersWrong)
                    
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
            

                
    print("FAILED FOLDS:", failedFolds)

    print('certain correct pretrained', certainCorrectPre)
    print('certain total pretrained', certainNumberPre)

    print('certain correct', certainCorrect)
    print('certain total', certainNumber)
    
    end = time.time()
    elapsed = end - start
    print(str(elapsed / 60), 'minutes elapsed.')

if __name__ == "__main__":
#    beforeAndAfter()
    plt.ion()
    justTreatmentGroups()
