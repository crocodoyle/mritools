import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics.pairwise import chi2_kernel

from sklearn.covariance import EmpiricalCovariance
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.svm import SVC
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib

import random

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

def calculateScores(predictions, actual):
    score = {}
        
    for scoreMet in scoringMetrics:
        score[scoreMet] = 0.0
    
    for predicted, actual in zip(predictions, actual):
        if predicted >= 0.5 and actual == 1:
            score['TP'] += 1.0
        elif predicted >= 0.5 and actual == 0:
            score['FP'] += 1.0
        elif predicted < 0.5 and actual == 0:
            score['TN'] += 1.0
        elif predicted < 0.5 and actual == 1:
            score['FN'] += 1.0
    
    try:
        score['sensitivity'] = score['TP'] / (score['TP'] + score['FN'])
    except:
        score['sensitivity'] = 0
     
    try:
        score['specificity'] = score['TN'] / (score['TN'] + score['FP'])
    except:
        score['specificity'] = 0

    return score
                

def countingClassifier(trainCounts, testCounts, trainOutcomes, testOutcomes):
    countingScore = {}
    
    for metric in metrics:
        nb = GaussianNB()
        nb.fit(trainCounts, trainOutcomes[metric])
        
        predictions = nb.predict(testCounts)
        
        countingScore[metric] = calculateScores(predictions, testOutcomes[metric])

    return (countingScore, predictions, nb)


def random_forest(trainData, testData, trainOutcomes, testOutcomes, mri_test, mixture_models, results_dir, rf=None):

    if rf == None:
        print('training random forest...')
        rf = RandomForestClassifier(class_weight='balanced', n_estimators=3000, n_jobs=-1)

        # for cross-validating tree depth
#            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
#            
#            for train_index, val_index in sss.split(trainData, trainOutcomes[metric]):
##                print 'training index', train_index
#                treeDepth = []
#                validationScore = []
#                for d in range(25, 30, 3):
#                    print 'cross validating tree depth:', d
#                    rf = RandomForestClassifier(class_weight='balanced', n_estimators=3000, n_jobs=-1, max_depth=d)
#                    rf.fit(np.asarray(trainData)[train_index], np.asarray(trainOutcomes[metric])[train_index])
#                    validationScore.append(rf.score(np.asarray(trainData)[val_index], np.asarray(trainOutcomes[metric])[val_index]))
#                    treeDepth.append(d)
#                    
#                bestDepth = treeDepth[np.argmin(validationScore)]
#
#                plt.plot(treeDepth, validationScore)
#                plt.title("RF Tree Depth Cross-Validation")
#                plt.xlabel("Max Depth Allowed in Trees")
#                plt.ylabel("Prediction Accuracy")
##                plt.show()
#                plt.close()
#
#            rf = RandomForestClassifier(class_weight='balanced', n_estimators=3000, n_jobs=-1, max_depth=bestDepth)
        rf.fit(trainData, trainOutcomes)
    else:
        print('Using previously trained model')

    predictions = rf.predict(testData)
    rfscore = calculateScores(predictions, testOutcomes)

    # correlation = []
    # for featNum in range(np.shape(trainData)[1]):
    #     correlation.append(stats.pearsonr(trainData[:, featNum], trainOutcomes)[0])
    #
    # x = np.linspace(1, len(rf.feature_importances_), num=len(rf.feature_importances_))
    # fig, (ax, ax2) = plt.subplots(1,2, sharex=True)
    # ax.bar(x, rf.feature_importances_)
    # ax.set_xlabel('Lesion-Type')
    # ax.set_ylabel('Gini Impurity (Normalized)')
    #
    # ax2.bar(x, np.abs(correlation))
    # ax2.set_xlabel('Lesion-Type')
    # ax2.set_ylabel('Pearson correlation coefficient (absolute value)')
    # plt.tight_layout()
    # plt.savefig(results_dir + 'feature-importance' + np.random.randint(10000) + '.png')
    # plt.close()

    endT = len(mixture_models[sizes[0]].weights_)
    endS = endT + len(mixture_models[sizes[1]].weights_)
    endM = endS + len(mixture_models[sizes[2]].weights_)
    endL = endM + len(mixture_models[sizes[3]].weights_)

    importance = {}
    importance['T'] = rf.feature_importances_[0:endT]
    importance['S'] = rf.feature_importances_[endT:endS]
    importance['M'] = rf.feature_importances_[endS:endM]
    importance['L'] = rf.feature_importances_[endM:endL]

    allImportances = rf.feature_importances_

    bestLesions, bestLesionsSize = [], []

    goodLesion = np.argmax(allImportances)
    allImportances[goodLesion] = 0

    if goodLesion >= endM:
        bestLesionsSize.append(3)
        bestLesions.append(goodLesion - endM)
    elif goodLesion >= endS:
        bestLesionsSize.append(2)
        bestLesions.append(goodLesion - endS)
    elif goodLesion >= endT:
        bestLesionsSize.append(1)
        bestLesions.append(goodLesion - endT)
    else:
        bestLesionsSize.append(0)
        bestLesions.append(goodLesion)

    # visualizing lesion groups
#        n = 10
#        for lesionNumber, (m, bestLesion) in enumerate(zip(bestLesionsSize, bestLesions)):
#            print lesionNumber, 'best lesion type:', sizes[m]
#            print bestLesion, importance[['T','S','M','L'][m]][bestLesion]
#            size = sizes[m]
#            typeNumber = 0
#            
#            for f1 in range(subtypeShape[m][1]):
#                for f2 in range(subtypeShape[m][2]):
#                    for f3 in range(subtypeShape[m][3]):
#                        for f4 in range(subtypeShape[m][4]):
#                            
#                            if typeNumber == bestLesion:
#                                plt.figure(figsize=(16,4))
#                                
#                                print len(lesionIndices[size][typeNumber]), 'lesions in group'
#                                print len(set(brainIndices[size][typeNumber])), 'brains in group'
#                                
#                                numActive = 0.0
#                                total = 0.0
#                                
#                                
#                                #TODO: FIX THIS
#                                try:
#                                    for index, b in enumerate(brainIndices[size][typeNumber]):
#                                        if b < len(mri_train):
#                                            if mri_train[b].treatment == "Placebo":
#                                                if trainOutcomes[b] == 1:
#                                                    numActive += 1.0
#                                                total += 1.0
#                                    
#                                    lesionPosterior = numActive / total
#                                    
#                                    
#                                    print 'P(A|T, LES' + str(lesionNumber) + ') =',  lesionPosterior
#                                except:
#                                    pass
#                                
##                                try:
#                                forVisualization = zip(brainIndices[size][typeNumber], lesionIndices[size][typeNumber])
#                                random.shuffle(forVisualization)
#                                brainIndices[size][typeNumber], lesionIndices[size][typeNumber] = zip(*forVisualization)
#                                i=0
#                                for (brainIndex, lesionIndex) in zip(brainIndices[size][typeNumber], lesionIndices[size][typeNumber]):
#                                    try:
#                                        scan = mri_train[brainIndex]
#                                        if scan.newT2 > 0:
#                                            pass
#            #                                    img = nib.load(scan.images['flr']).get_data()
#                                        img = nib.load(scan.images['t2w']).get_data()
#                                        lesionMaskImg = np.zeros((np.shape(img)))
#                                        
#            #                                for lesion in scan.lesionList:
#            #                                    for point in lesion:
#            #                                        lesionMaskImg[point[0], point[1], point[2]] = 1
#                                        
#                                        for point in scan.lesionList[lesionIndex]:
#                                            lesionMaskImg[point[0], point[1], point[2]] = 1
#                                        
#                                        x, y, z = [int(np.mean(xxx)) for xxx in zip(*scan.lesionList[lesionIndex])]
#                            
#                                        maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)                      
#                                        maskSquare = np.zeros((np.shape(img)))
#                                        maskSquare[x-10:x+10, y+10, z] = 1
#                                        maskSquare[x-10:x+10, y-10, z] = 1
#                                        maskSquare[x-10, y-10:y+10, z] = 1
#                                        maskSquare[x+10, y-10:y+10, z] = 1
#                                                      
#                                        square = np.ma.masked_where(maskSquare == 0, np.ones(np.shape(maskSquare))*5000)
#                               
#                                        lesionMaskPatch = maskImg[x-20:x+20, y-20:y+20, z]
#                                        ax = plt.subplot(2, n, i+1)
#                                        ax.set_xticks([])
#                                        ax.set_yticks([])
#                                        ax.imshow(img[20:200,20:200, z].T, cmap = plt.cm.gray, interpolation = 'nearest',origin='lower')
#                                        ax.imshow(maskImg[20:200,20:200, z].T, cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
#                                        ax.imshow(square[20:200, 20:200, z].T, cmap = plt.cm.autumn, interpolation = 'nearest', origin='lower')
#                                        
#                                        if scan.newT2 > 0:
#                                            ax.set_xlabel("Active Patient")
#                                        else:
#                                            ax.set_xlabel("Inactive Patient")
#                                        
#                                        ax3 = plt.subplot(2, n, i+1+n)
#                                        ax3.imshow(img[x-20:x+20, y-20:y+20, z].T, cmap = plt.cm.gray, interpolation = 'nearest', origin='lower')
#                                        ax3.imshow(lesionMaskPatch.T, cmap = plt.cm.autumn, alpha = 0.4, interpolation = 'nearest', origin='lower')
#                                        ax3.axes.get_yaxis().set_visible(False)
#                                        ax3.set_xticks([])
#                                        ax3.set_xlabel(letters[i])
#                                        i += 1
#                                        print scan.uid, '[', x, y, z, ']'
#                                    except:
##                                        print "problem with one of the images"
#                                        i -= 1
#                                    
#                                    if i == n:
#                                        break
#                                   
#        
#                                plt.subplots_adjust(wspace=0.01,hspace=0.01)
#                                plt.savefig('/usr/local/data/adoyle/images/lesions-'+ size + '-' + ''.join((str(f1),str(f2),str(f3),str(f4))) + '.png', dpi=500)
#                                plt.show()
##                                except:
##                                    print 'something went wrong for this lesion-type'
##                                    print brainIndices[size][typeNumber]
#                                
#                            typeNumber += 1

    # Predict only high-probability cases
    probabilities = rf.predict_proba(testData)

    probPredicted, actual = [], []
    certainCorrect, certainTotal = 0, 0

    for prob, outcome, scan, bol_rep in zip(probabilities, testOutcomes, mri_test, testData):
        # print(prob, outcome)
        if prob[0] > 0.8 and outcome == 0:
            probPredicted.append(0)
            actual.append(0)

            img = nib.load(scan.images['t2w']).get_data()
            lesionMaskImg = np.zeros((np.shape(img)))

            for lesion in scan.lesionList:
                for point in lesion:
                    lesionMaskImg[point[0], point[1], point[2]] = 1

            maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)

            fig = plt.figure(figsize=(3,6))
            ax = fig.add_subplot(2, 1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img[30, 20:225, 20:150], cmap = plt.cm.gray, interpolation = 'nearest', origin='lower')
            ax.imshow(maskImg[30, 20:225, 20:150], cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
            ax.set_xlabel('High probability inactive')

            ax = fig.add_subplot(2, 1, 2)
            x = np.linspace(1, len(bol_rep), num=len(bol_rep))
            ax.bar(x, bol_rep)
#                ax.set_ylabel('Number of Lesions')
            ax.set_xlabel('Lesion-Types')

            plt.savefig(results_dir + 'inactive-' + scan.uid, dpi=500)
            plt.close()

            certainCorrect += 1
            certainTotal += 1

        if prob[0] > 0.8 and outcome == 1:
            probPredicted.append(0)
            actual.append(1)

            certainTotal += 1

        if prob[1] > 0.8 and outcome == 1:
            probPredicted.append(1)
            actual.append(1)

            img = nib.load(scan.images['t2w']).get_data()
            lesionMaskImg = np.zeros((np.shape(img)))

            for lesion in scan.lesionList:
                for point in lesion:
                    lesionMaskImg[point[0], point[1], point[2]] = 1

            maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg)))*5000)

            fig = plt.figure(figsize=(3.5,6))
            ax = fig.add_subplot(2, 1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img[30, 20:225, 20:150], cmap = plt.cm.gray, interpolation = 'nearest',origin='lower')
            ax.imshow(maskImg[30, 20:225, 20:150], cmap = plt.cm.autumn, interpolation = 'nearest', alpha = 0.4, origin='lower')
            ax.set_xlabel('High probability active')

            ax = fig.add_subplot(2, 1, 2)
            x = np.linspace(1, len(bol_rep), num=len(bol_rep))
            ax.bar(x, bol_rep)
            ax.set_xlabel('Lesion-Types')

            plt.savefig(results_dir + 'active-' + scan.uid, dpi=500)
            plt.close()

            certainCorrect += 1
            certainTotal += 1

        if prob[1] > 0.8 and outcome == 0:
            probPredicted.append(1)
            actual.append(0)

            certainTotal += 1

    onlyCertainScore = calculateScores(probPredicted, actual)
        
    return (rfscore, predictions, rf), (onlyCertainScore, probabilities), (certainCorrect, certainTotal)


def identify_responders(trainData, testData, trainOutcomes, testOutcomes, train_patients, test_patients, drug_rf, placebo_rf):
    relapse_certainty = 0.8

    train_activity = placebo_rf.predict_proba(trainData)
    test_activity = placebo_rf.predict_proba(testData)
    
    responder_label_train = np.zeros((len(trainOutcomes)), dtype='bool')
    responder_label_test  = np.zeros((len(testOutcomes)), dtype='bool')

    responder_train_weight = np.zeros((len(trainOutcomes)), dtype='float')

    # BoL RF responder setup
    for index, (prediction, actual) in enumerate(zip(train_activity, trainOutcomes)):
        #predicted active but actually inactive
        if prediction[1] > relapse_certainty and actual == 0:
            responder_label_train[index] = 1
            responder_train_weight[index] = prediction[1]
        else:
            responder_label_train[index] = 0
            responder_train_weight[index] = prediction[0]
    
    for index, (prediction, actual) in enumerate(zip(test_activity, testOutcomes)):
        if prediction[1] > relapse_certainty and actual == 0:
            responder_label_test[index] = 1
        else:
            responder_label_test[index] = 0
            
    print('training responders:', np.sum(responder_label_train))
    print('training non-responders:', (len(trainOutcomes) - np.sum(responder_label_train)))
    
    print('testing responders:', np.sum(responder_label_test))
    print('testing non-responders:', (len(testOutcomes) - np.sum(responder_label_test)))

    predictions = drug_rf.predict_proba(testData)
    
    responder_predictions = []    
    
    for index, (prediction, actual) in enumerate(zip(predictions, responder_label_test)):
        if prediction[1] > 0.5:
            responder_predictions.append(1)
        if prediction[0] > 0.5:
            responder_predictions.append(0)
    
    responder_score = calculateScores(responder_predictions, responder_label_test)
    

    high_prob_responder_predictions = []
    high_prob_responder_actual = []

    for index, (prediction, actual) in enumerate(zip(predictions, responder_label_test)):
        print('high prob responder predictions:', prediction, actual)
        if prediction[1] > 0.8:
            high_prob_responder_predictions.append(1)
            high_prob_responder_actual.append(actual)
        if prediction[0] > 0.8:
            high_prob_responder_predictions.append(0)
            high_prob_responder_actual.append(actual)
    
    print('high probability predictions:', len(high_prob_responder_predictions))
    high_prob_scores = calculateScores(high_prob_responder_predictions, high_prob_responder_actual)

    return (responder_score, responder_predictions), high_prob_scores


def svms(trainData, testData, trainOutcomes):
    linear = SVC(kernel='linear', class_weight='balanced', probability=True)
    linear.fit(trainData, trainOutcomes)
    svm_linear_posterior = linear.predict_proba(testData)

    rbf = SVC(class_weight='balanced', probability=True)
    rbf.fit(trainData, trainOutcomes)
    svm_rbf_posterior = rbf.predict_proba(testData)

    trainDistances = chi2_kernel(trainData, trainData)
    testDistances = chi2_kernel(testData, trainData)

    svc = SVC(kernel='precomputed', class_weight='balanced', probability=True)
    svc.fit(trainDistances, trainOutcomes)

    chi2svm_posterior = svc.predict_proba(testDistances)

    return svm_linear_posterior, svm_rbf_posterior, chi2svm_posterior


def knn(trainData, trainOutcomes, testData):

    try:
        knnEuclidean = KNeighborsClassifier(n_neighbors=1)
        knnEuclidean.fit(trainData, trainOutcomes)
        knn_euclid_posterior = knnEuclidean.predict_proba(testData)
    except np.linalg.linalg.LinAlgError as e:
        knn_euclid_posterior = np.zeros((len(trainOutcomes), 2))
        knn_euclid_posterior[:, 1] = 1
        print('Not enough samples for Euclidean covariance estimation! Predicting all active.')
        print(e)
    try:
        knnMahalanobis = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric = 'mahalanobis')
        knnMahalanobis.fit(trainData, trainOutcomes)
        knn_maha_posterior = knnMahalanobis.predict_proba(testData)
    except np.linalg.linalg.LinAlgError as e:
        print('Not enough samples for Mahalanobis covariance estimation! Predicting all active.')
        print(e)
        knn_maha_posterior = np.zeros((len(trainOutcomes), 2))
        knn_maha_posterior[:, 1] = 1

    return knn_euclid_posterior, knn_maha_posterior


def randomForestFeatureSelection(trainData, testData, trainOutcomes, testOutcomes, minTypes):
    train = trainData
    test = testData    
    
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=2000, n_jobs=-1, oob_score=True)
    rf.fit(trainData, trainOutcomes)
    
    allFeatureImportance = rf.feature_importances_
    featureImportance = allFeatureImportance
 
    typesLeft = len(featureImportance)
    
    trainScores, oobScores, testScores, numFeatures = [], [], [] ,[]

    while True:
        removeThisRound = []
        
        for r in range(int(np.ceil(0.2*typesLeft))):
            remove = np.argmin(featureImportance)
            removeThisRound.append(remove)
            
            featureImportance[remove] = 999
        
        featureImportance = [x for x in featureImportance if x != 999]
        
        removeThisRound = sorted(removeThisRound, reverse=True)
        
        for remove in removeThisRound:
            train = np.delete(train, remove, 1)
            test = np.delete(test, remove, 1)

        typesLeft -= len(removeThisRound)
        
        rf.fit(train, trainOutcomes)
        
        trainScores.append(rf.score(train, trainOutcomes))
        oobScores.append(rf.oob_score_)
        testScores.append(rf.score(test, testOutcomes))
        numFeatures.append(typesLeft)
        
        if typesLeft < minTypes:
            break

    print(numFeatures[np.argmax(oobScores)], 'is the optimal number of features')
    
    plt.figure()
    plt.plot(numFeatures, oobScores, label="Out-of-Bag Score")
    plt.plot(numFeatures, trainScores, label="Training Score")
    plt.plot(numFeatures, testScores, label="Test Score")
    plt.xlabel('Number of codewords in BoL')
    plt.ylabel('Mean Accuracy')
    plt.legend()
    plt.tight_layout()
#        plt.show()
        
    train = trainData
    test = testData
    finalRemove = np.shape(testData)[1] - numFeatures[np.argmax(oobScores)]
    
    removeThisRound = []
    for r in range(finalRemove):
        remove = np.argmin(allFeatureImportance)
        removeThisRound.append(remove)
        
        allFeatureImportance[remove] = 999
        
    # this is where I need to remove/update the visualization arrays
    allFeatureImportance = [x for x in featureImportance if x != 999]
    
    removeThisRound = sorted(removeThisRound, reverse=True)
    
    for remove in removeThisRound:
        train = np.delete(train, remove, 1)
        test = np.delete(test, remove, 1)
    
        
    return train, test, removeThisRound

