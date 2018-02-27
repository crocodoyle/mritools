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

scoringMetrics = ['TP', 'FP', 'TN', 'FN']

lbpRadii = [1,2,3]
riftRadii = [1,2,3]
selectK = False
visualizeAGroup = False

letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)']

treatments = ['Placebo', 'Laquinimod', 'Avonex']
treatment_labels = ['Untreated', 'Drug A', 'Drug B']

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


def random_forest(trainData, testData, trainOutcomes, rf=None):

    if rf == None:
        print('training random forest...')
        rf = RandomForestClassifier(class_weight='balanced', n_estimators=3000, n_jobs=-1)
        rf.fit(trainData, trainOutcomes)
    else:
        print('Using previously trained model')

    predictions = rf.predict(testData)
    probabilities = rf.predict_proba(testData)
        
    return predictions, rf, probabilities


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

