
import os, pickle, time
from collections import defaultdict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.mixture import GMM
from sklearn.model_selection import StratifiedKFold

import load_data
import bol_classifiers
from analyze_lesions import createRepresentationSpace, testRepresentationSpace, separatePatientsByTreatment, removeWorstFeatures, showWhereTreatmentHelped, plotScores

from mri import mri

treatments = ['Placebo', 'Laquinimod', 'Avonex']
modalities = ['t1p', 't2w', 'pdw', 'flr']
tissues = ['csf', 'wm', 'gm', 'pv', 'lesion']

feats = ["Context", "RIFT", "LBP", "Intensity"]
sizes = ["tiny", "small", "medium", "large"]

scoringMetrics = ['TP', 'FP', 'TN', 'FN']

metrics = ['newT2']


workdir = '/home/users/adoyle/respondMS/'
datadir = '/data1/users/adoyle/MS-LAQ/MS-LAQ-302-STX/'

mri_list_location = workdir + 'mri_list.pkl'



def predict_responders():
    start = time.time()

    try:
        experiment_number = pickle.load(open(workdir + 'experiment_number.pkl', 'rb'))
        experiment_number += 1
    except:
        print('Couldnt find the file to load experiment number')
        experiment_number = 0

    print('This is experiment number:', experiment_number)

    results_dir = workdir + '/experiment-' + str(experiment_number) + '/'
    os.makedirs(results_dir)

    pickle.dump(experiment_number, open(workdir + 'experiment_number.pkl', 'wb'))

    mri_list = pickle.load(open(mri_list_location, 'rb'))
    mri_list, without_clinical = load_data.loadClinical(mri_list)

    outcomes = load_data.get_outcomes(mri_list)

    kf = StratifiedKFold(50, shuffle=True, random_state=42)

    respondersRight, respondersWrong = {}, {}
    failedFolds = 0

    certainNumber, certainCorrect, certainNumberPre, certainCorrectPre = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)

    scores = defaultdict(dict)

    knnEuclideanScores, knnMahalanobisScores, chi2Scores, chi2svmScores, featureScores, svmLinScores, svmRadScores, preTrainedFeatureScores, preTrainedSvmLinScores, preTrainedSvmRadScores = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    countingScores = defaultdict(dict)

    bestScores, bestKnnEuclideanScores, bestKnnMahalanobisScores, bestChi2Scores, bestChi2svmScores, bestFeatureScores, bestSvmLinScores, bestSvmRadScores, bestPreTrainedKnnEuclideanScores, bestPreTrainedFeatureScores, bestPreTrainedSvmLinScores, bestPreTrainedSvmRadScores = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    probScores, allProbScores = defaultdict(dict), defaultdict(dict)

    responderScores, responderHighProbScores, countScores = defaultdict(dict), defaultdict(dict), defaultdict(dict)

    r1, r2, r3, r4 = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)

    for treatment in treatments:
        scores[treatment] = defaultdict(list)
        knnEuclideanScores[treatment] = defaultdict(list)
        knnMahalanobisScores[treatment] = defaultdict(list)
        chi2Scores[treatment] = defaultdict(list)
        chi2svmScores[treatment] = defaultdict(list)
        featureScores[treatment] = defaultdict(list)
        svmLinScores[treatment] = defaultdict(list)
        svmRadScores[treatment] = defaultdict(list)
        preTrainedFeatureScores[treatment] = defaultdict(list)
        bestPreTrainedSvmLinScores[treatment] = defaultdict(list)
        bestPreTrainedSvmRadScores[treatment] = defaultdict(list)
        countingScores[treatment] = defaultdict(list)
        bestScores[treatment] = defaultdict(list)
        bestKnnEuclideanScores[treatment] = defaultdict(list)
        bestKnnMahalanobisScores[treatment] = defaultdict(list)
        bestChi2Scores[treatment] = defaultdict(list)
        bestChi2svmScores[treatment] = defaultdict(list)
        bestFeatureScores[treatment] = defaultdict(list)
        bestSvmLinScores[treatment] = defaultdict(list)
        bestSvmRadScores[treatment] = defaultdict(list)
        bestPreTrainedKnnEuclideanScores[treatment] = defaultdict(list)
        bestPreTrainedFeatureScores[treatment] = defaultdict(list)
        preTrainedSvmLinScores[treatment] = defaultdict(list)
        preTrainedSvmRadScores[treatment] = defaultdict(list)
        probScores[treatment], allProbScores[treatment] = defaultdict(list), defaultdict(list)

        responderScores[treatment], responderHighProbScores[treatment], countScores[treatment] = defaultdict(
            list), defaultdict(list), defaultdict(list)

        certainNumber[treatment], certainCorrect[treatment], certainNumberPre[treatment], certainCorrectPre[
            treatment] = 0, 0, 0, 0
        respondersRight[treatment], respondersWrong[treatment] = 0, 0

        r1[treatment], r2[treatment], r3[treatment], r4[treatment] = defaultdict(list), defaultdict(list), defaultdict(
            list), defaultdict(list)

    # initialization of result structures complete
    # start learning BoL, predicting activity
    for foldNum, (train_index, test_index) in enumerate(kf.split(range(len(mri_list)), outcomes['newT2'])):
        print(foldNum+1, '/', len(kf)+1)
        scoreThisFold = True

        mri_train, mri_test = np.asarray(mri_list)[train_index], np.asarray(mri_list)[test_index]
        trainCounts, testCounts = load_data.loadLesionNumbers(mri_train), load_data.loadLesionNumbers(mri_test)

        # incorporate patients with no clinical data
        train_patients = []
        for scan in mri_train:
            train_patients.append(scan)
        for scan in without_clinical:
            train_patients.append(scan)

        print('loading feature data...')
        startLoad = time.time()
        numLesionsTrain, lesionSizesTrain, lesionCentroids, brainUids = load_data.getLesionSizes(train_patients)
        trainDataVectors, lbpPCA = load_data.loadAllData(train_patients, numLesionsTrain)

        numLesionsTest, lesionSizesTest, lesionCentroids, brainUids = load_data.getLesionSizes(mri_test)
        dataVectorsTest, lbpPCA = load_data.loadAllData(mri_test, numLesionsTest, lbpPCA=lbpPCA)

        print('loading data took', (time.time() - startLoad) / 60.0, 'minutes')

        print('removing infrequent features...')
        startPruneTime = time.time()
        prunedDataTrain = []
        prunedDataTest = []

        for dTrain, dTest in zip(trainDataVectors, dataVectorsTest):
            dTrainPruned, dTestPruned = load_data.prune_features(dTrain, dTest)
            prunedDataTrain.append(dTrainPruned)
            prunedDataTest.append(dTestPruned)

        print("it took", (time.time() - startPruneTime) / 60.0, "minutes")
        print('learning bag of lesions...')

        startBol = time.time()
        allTrainData, clusters, pcas, subtypeShape, brainIndices, lesionIndices = createRepresentationSpace(
            train_patients, prunedDataTrain, lesionSizesTrain, len(mri_train), lesionCentroids, examineClusters=False)
        elapsedBol = time.time() - startBol
        print(str(elapsedBol / 60), 'minutes to learn BoL.')

        print('transforming test data to bag of lesions representation...')
        allTestData = testRepresentationSpace(mri_test, prunedDataTest, lesionSizesTest, clusters, pcas)

        trainingPatientsByTreatment, testingPatientsByTreatment, trainingData, testingData, trainCounts, testCounts = separatePatientsByTreatment(
            mri_train, mri_test, allTrainData, allTestData, trainCounts, testCounts)

        # feature selection
        featuresToRemove, c = None, None
        for treatment in treatments:
            try:

                trainData, testData = trainingData[treatment], testingData[treatment]
                trainDataCopy, testDataCopy = trainData, testData
                trainOutcomes, testOutcomes = load_data.get_outcomes(trainingPatientsByTreatment[treatment]), load_data.get_outcomes(
                    testingPatientsByTreatment[treatment])

                remove_worst_features = True
                if remove_worst_features:
                    if treatment == "Placebo":
                        print('selecting features...')
                        bestTrainData, bestTestData, featuresToRemove = bol_classifiers.randomForestFeatureSelection(
                            trainDataCopy, testDataCopy, trainOutcomes['newT2'], testOutcomes['newT2'], 12)
                    else:
                        bestTrainData, bestTestData = removeWorstFeatures(trainDataCopy, testDataCopy, featuresToRemove)
                else:
                    bestTrainData = trainDataCopy
                    bestTestData = testDataCopy

                if treatment == "Placebo":
                    (bestFeatureScore, bestFeaturePredictions, placebo_rf), (probScore, probPredicted), (
                    correct, total) = bol_classifiers.featureClassifier(bestTrainData, bestTestData, trainOutcomes,
                                                                        testOutcomes, subtypeShape, train_patients,
                                                                        mri_test, brainIndices, lesionIndices,
                                                                        len(mri_list))

                    (bestChi2Score, bestChi2Predictions), (
                    bestChi2svmscore, bestChi2svmPredictions) = bol_classifiers.chi2Knn(bestTrainData, bestTestData,
                                                                                        trainOutcomes, testOutcomes)
                    (bestSvmLinScore, bestSvmLinPredictions, svm1), (
                    bestSvmRadScore, bestSvmRadPredictions, svm2) = bol_classifiers.svmClassifier(bestTrainData,
                                                                                                  bestTestData,
                                                                                                  trainOutcomes,
                                                                                                  testOutcomes)
                    (bestKnnEuclideanScoreVals, bestEuclideanPredictions), (
                    bestKnnMahalanobisScoreVals, bestMahalanobisPredictions) = bol_classifiers.knn(bestTrainData,
                                                                                                   trainOutcomes,
                                                                                                   bestTestData,
                                                                                                   testOutcomes)

                    (featureScore, featurePredictions, meh), (allProbScore, allprobPredicted), (
                    allCorrect, allTotal) = bol_classifiers.featureClassifier(trainData, testData, trainOutcomes,
                                                                              testOutcomes, subtypeShape,
                                                                              train_patients, mri_test, brainIndices,
                                                                              lesionIndices, len(mri_list))

                    (countingScore, countingPredictions, placebo_nb) = bol_classifiers.countingClassifier(
                        trainCounts[treatment], testCounts[treatment], trainOutcomes, testOutcomes)

                # drugged patients
                else:
                    # natural course MS model
                    (bestPreTrainedFeatureScore, bestPreTrainedFeaturePredictions, meh), (
                    pretrainedProbScore, pretrainedProbPredicted), (correct, total) = bol_classifiers.featureClassifier(
                        bestTrainData, bestTestData, trainOutcomes, testOutcomes, subtypeShape, train_patients,
                        mri_test, brainIndices, lesionIndices, len(mri_list), placebo_rf)

                    # new model on drugged patients
                    (bestFeatureScore, bestFeaturePredictions, meh), (probScore, probDrugPredicted), (
                    correct, total) = bol_classifiers.featureClassifier(bestTrainData, bestTestData, trainOutcomes,
                                                                        testOutcomes, subtypeShape, train_patients,
                                                                        mri_test, brainIndices, lesionIndices,
                                                                        len(mri_list))

                    certainNumber[treatment] += total
                    certainCorrect[treatment] += correct

                    right, wrong, r1_score, r2_score, r3_score, r4_score = showWhereTreatmentHelped(
                        pretrainedProbPredicted, probDrugPredicted, bestTrainData, bestTestData, trainOutcomes['newT2'],
                        testOutcomes['newT2'], trainingPatientsByTreatment[treatment],
                        testingPatientsByTreatment[treatment])

                    respondersRight[treatment] += right
                    respondersWrong[treatment] += wrong

                    print('responders right', respondersRight)
                    print('responders wrong', respondersWrong)

                    (responderScore,
                     responderProbs), responderHighProbScore, count_score = bol_classifiers.identifyResponders(
                        bestTrainData, bestTestData, trainOutcomes, testOutcomes, trainCounts[treatment],
                        testCounts[treatment], placebo_rf, placebo_nb)

                certainNumberPre[treatment] += total
                certainCorrectPre[treatment] += correct

                for scoreMet in scoringMetrics + ['sensitivity', 'specificity']:
                    featureScores[treatment][scoreMet].append(featureScore['newT2'][scoreMet])

                    # bad classifiers
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

                for treatment in treatments:
                    if treatment == "Avonex":
                        #                plotScores([(responderScores[treatment], 'Responders'), (responderHighProbScores[treatment], 'Responders (certain)'), (countScores[treatment], 'Responders (lesion counts)')], "Avonex Responder Prediction")
                        plotScores([(r1[treatment], 'Responders'), (r2[treatment], 'Responders (certain GT)'),
                                    (r3[treatment], 'Responders (certain prediction)'),
                                    (r4[treatment], 'Responders (all certain)')], "Avonex Responder Prediction")
                    elif treatment == "Laquinimod":
                        #                plotScores([(responderScores[treatment], 'Responders'), (responderHighProbScores[treatment], 'Responders (certain)'), (countScores[treatment], 'Responders (lesion counts)')], "Laquinimod Responder Prediction")
                        plotScores([(r1[treatment], 'Responders'), (r2[treatment], 'Responders (certain GT)'),
                                    (r3[treatment], 'Responders (certain prediction)'),
                                    (r4[treatment], 'Responders (all certain)')], "Laquinimod Responder Prediction")

                bestScoring = []

                for treatment in treatments:

                    if treatment == "Placebo":
                        bestScoring.append((bestFeatureScores[treatment], 'Untreated ($\\alpha=0.5$)'))
                        bestScoring.append((probScores[treatment], 'Untreated ($\\alpha=0.8$)'))

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
    predict_responders()