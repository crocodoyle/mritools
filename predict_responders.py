
import os, pickle, time, csv
from collections import defaultdict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.mixture import GMM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

import load_data
import bol_classifiers
from analyze_lesions import learn_bol, project_to_bol, separatePatientsByTreatment, removeWorstFeatures, showWhereTreatmentHelped, plotScores

from mri import mri

treatments = ['Placebo', 'Laquinimod', 'Avonex']
treatment_labels = ['Placebo', 'Drug A', 'Drug B']

classifier_names = ['Random Forest', 'SVM (Linear)', 'SVM (RBF)', '1-NN ($\\chi^2$)', '1-NN (Mahalanobis)']

modalities = ['t1p', 't2w', 'pdw', 'flr']
tissues = ['csf', 'wm', 'gm', 'pv', 'lesion']

feats = ["Context", "RIFT", "LBP", "Intensity"]
sizes = ["tiny", "small", "medium", "large"]

scoringMetrics = ['TP', 'FP', 'TN', 'FN']

metrics = ['newT2']

datadir = '/data1/users/adoyle/MS-LAQ/MS-LAQ-302-STX/'

mri_list_location = datadir + 'mri_list.pkl'

def responder_roc(activity_truth, activity_posterior, untreated_posterior, results_dir):

    fig1 = plt.figure(0)
    ax = fig1.add_subplot(1, 1, 1)
    for treatment in treatments:
        p_a_auc, p_d_distance, p_d_harmonic_mean, p_d_anti_harmonic_mean = [], [], [], []
        p_a_brier = []

        if 'Placebo' not in treatment:
            a_prob = np.concatenate(tuple(untreated_posterior[treatment]), axis=0) # must use predictions for untreated
            a_prob = a_prob[:, 1]                                                  # to infer what would have happened if untreated

            # print('Untreated predictions (' + treatment + '):', a_prob)

            d_true = np.concatenate(tuple(activity_truth[treatment]), axis=0)
            d_prob = np.concatenate(tuple(activity_posterior[treatment]), axis=0)

            d_prob = d_prob[:, 1]                          # just consider the P(A=1|BoL, treatment)

            a_range = np.linspace(0, 1, 50, endpoint=False)
            d_range = np.linspace(0, 1, 50, endpoint=False)

            fig2 = plt.figure(1, figsize=(10, 10))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))

            ax1.plot(a_range, a_range, "k:", label="Perfectly calibrated")

            for p_a in a_range:
                try:
                    a_true_inferred = np.zeros(a_prob.shape)
                    a_true_inferred[a_prob > p_a] = 1

                    # print('A untreated predictions:', a_true_inferred)
                    fraction_of_positives, mean_predicted_value = calibration_curve(a_true_inferred, a_prob, n_bins=10)

                    score = brier_score_loss(a_true_inferred, a_prob)
                    p_a_brier.append(score)

                    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (str(p_a), score))

                    # tn, tp, _ = roc_curve(a_true_inferred, a_prob)
                    auc_weighted = roc_auc_score(a_true_inferred, a_prob, 'weighted')
                    auc_macro = roc_auc_score(a_true_inferred, a_prob, 'macro')
                    auc_micro = roc_auc_score(a_true_inferred, a_prob, 'micro')
                    auc_samples = roc_auc_score(a_true_inferred, a_prob, 'samples')

                    # print('AUCs (weighted, macro, micro, smaples):', auc_weighted, auc_macro, auc_micro, auc_samples)

                    p_a_auc.append(auc_macro)
                except:
                    print('AUC undefined for:', p_a)
                    p_a_auc.append(0)
                    p_a_brier.append(1)

            ax2.hist(a_prob, range=(0, 1), bins=10, label='P(A|BoL, untr)', histtype="step", lw=2)

            ax1.set_ylabel("Fraction of positives")
            ax1.set_ylim([-0.05, 1.05])
            ax1.legend(loc="lower right", shadow=True)
            ax1.set_title('Calibration plots  (reliability curve)')

            ax2.set_xlabel("Mean predicted value")
            ax2.set_ylabel("Count")
            ax2.legend(loc="upper center", ncol=2, shadow=True)

            fig2.tight_layout()
            fig2.savefig(results_dir, treatment + '_calibration_curve.png')

            best_p_a = a_range[np.argmin(p_a_brier)]
            a_true = np.ones(a_prob.shape)
            a_true[a_prob <= best_p_a] = 0

            print('P(A|BoL, untr) Brier scores: ', p_a_brier)
            print('Best theshold:', best_p_a)

            for p_d in d_range:
                try:
                    d_predicted = np.zeros(d_prob.shape)
                    d_predicted[d_prob <= p_d] = 0
                    d_predicted[d_prob > p_d] = 1

                    tn, fp, fn, tp = confusion_matrix(d_true, d_predicted).ravel()

                    sens = tp/(tp + fn)
                    spec = tn/(tn + fp)

                    distance = np.sqrt( (1 - sens)**2 + (1 - spec)**2 )
                    harmonic_mean = 2*sens*spec / (sens + spec)
                    anti_harmonic_mean = sens * spec / (2 - sens*spec)

                    p_d_distance.append(distance)
                    p_d_harmonic_mean.append(harmonic_mean)
                    p_d_anti_harmonic_mean.append(anti_harmonic_mean)
                except:
                    print('sens/spec or something else undefined for', p_d)
                    p_d_distance.append(1)
                    p_d_harmonic_mean.append(0)
                    p_d_anti_harmonic_mean.append(0)

            print('P(A|BoL, ' + treatment + ') sensitivity/specificity harmonic means: ', p_d_harmonic_mean)

            # select operating point with best AUC

            best_p_d = d_range[np.argmax(p_d_harmonic_mean)]

            # best is min distance, max (anti) harmonic mean of sens/spec
            print('Best P(A|BoL, ' + treatment + ') using distance:', d_range[np.argmin(p_d_distance)])
            print('Best P(A|BoL, ' + treatment + ') using harmonic mean:', d_range[np.argmax(p_d_harmonic_mean)])
            print('Best P(A|BoL, ' + treatment + ') using anti-harmonic mean:', d_range[np.argmax(p_d_anti_harmonic_mean)])

            print('Best threshold for untreated activity prediction: ', best_p_a)
            print('Best threshold for treated activity prediction: ', best_p_d)

            d_predicted = np.zeros(d_prob.shape)
            d_predicted[d_prob <= best_p_d] = 0
            d_predicted[d_prob > best_p_d] = 1

            r_true = np.zeros(a_true.shape)          # Assumption that our Placebo future lesion activity classifier is perfect
            r_true[d_true == 0] = 1                  # and that responders have no future lesion activity on drug
            r_true[a_true == 0] = 0

            r_predicted = np.zeros(d_predicted.shape)     # Responders are predicted when active on Placebo
            r_predicted[a_true == 1] = 1                  # and inactive on the drug
            r_predicted[d_predicted > best_p_d] = 0

            roc_auc = roc_auc_score(r_true, r_predicted, 'weighted')
            fpr, tpr, _ = roc_curve(r_true, r_predicted)

            lw = 2
            if 'Laquinimod' in treatment:
                ax.plot(fpr, tpr, color='darkorange', lw=lw, label=treatment + ' ROC (AUC = %0.2f)' % roc_auc)
                ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            else:
                ax.plot(fpr, tpr, color='darkred', lw=lw, label=treatment + ' ROC (AUC = %0.2f)' % roc_auc)

            # plt.title('Receiver operating characteristic example', fontsize=24)

            print(treatment + ' optimal thresholds (activity, drug_activity): ', best_p_a, best_p_d)

    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True, fontsize=20)
    fig1.savefig(results_dir + 'responder_' + 'p_a_'+ str(best_p_a) + '_p_d_' + str(best_p_d) + '_roc.png', bbox_inches='tight')

    return best_p_a, best_p_d

def predict_responders():
    start = time.time()

    try:
        experiment_number = pickle.load(open(datadir + 'experiment_number.pkl', 'rb'))
        experiment_number += 1
    except:
        print('Couldnt find the file to load experiment number')
        experiment_number = 0

    print('This is experiment number:', experiment_number)

    results_dir = datadir + '/experiment-' + str(experiment_number) + '/'
    os.makedirs(results_dir)

    pickle.dump(experiment_number, open(datadir + 'experiment_number.pkl', 'wb'))

    mri_list = pickle.load(open(mri_list_location, 'rb'))
    mri_list, without_clinical = load_data.loadClinical(mri_list)

    print('We have ' + str(len(mri_list)) + ' patients who finished the study and ' + str(len(without_clinical)) + ' who did not')
    outcomes = load_data.get_outcomes(mri_list)

    with open(results_dir + 'responders.csv', 'w') as responder_file:
        responder_writer = csv.writer(responder_file)
        responder_writer.writerow(['Subject ID', 'Treatment', '# T2 Lesions', 'P(A=1|BoL, untr)', 'P(A=0|BoL, tr'])

        patient_results = {}
        for scan in mri_list:
            patient_results[scan.uid] = {}

        kf = StratifiedKFold(50, shuffle=True, random_state=42)

        respondersRight, respondersWrong = {}, {}
        failedFolds = 0

        certainNumber, certainCorrect, certainNumberPre, certainCorrectPre = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)

        scores = defaultdict(dict)
        activity_posterior, activity_truth, untreated_posterior = defaultdict(list), defaultdict(list), defaultdict(list)
        euclidean_knn_posterior, mahalanobis_knn_posterior, chi2_svm_posterior, rbf_svm_posterior, linear_svm_posterior, naive_bayes_posterior =  defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

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
        for foldNum, (train_index, test_index) in enumerate(kf.split(range(len(mri_list)), outcomes)):
            print(foldNum+1, '/', kf.get_n_splits())
            scoreThisFold = True

            mri_train, mri_test = np.asarray(mri_list)[train_index], np.asarray(mri_list)[test_index]
            trainCounts, testCounts = load_data.loadLesionNumbers(mri_train), load_data.loadLesionNumbers(mri_test)

            # incorporate patients with no clinical data
            train_patients = []
            for scan in mri_train:
                train_patients.append(scan)
            for scan in without_clinical:
                train_patients.append(scan)

            # print('loading feature data...')
            # startLoad = time.time()
            numLesionsTrain, lesionSizesTrain, lesionCentroids, brainUids = load_data.getLesionSizes(train_patients)
            trainDataVectors, lbpPCA = load_data.loadAllData(train_patients, numLesionsTrain)

            numLesionsTest, lesionSizesTest, lesionCentroids, brainUids = load_data.getLesionSizes(mri_test)
            dataVectorsTest, lbpPCA = load_data.loadAllData(mri_test, numLesionsTest, lbpPCA=lbpPCA)

            # print('loading data took', (time.time() - startLoad) / 60.0, 'minutes')

            # print('removing infrequent features...')
            # startPruneTime = time.time()
            # prunedDataTrain = []
            # prunedDataTest = []
            #
            # for dTrain, dTest in zip(trainDataVectors, dataVectorsTest):
            #     dTrainPruned, dTestPruned = load_data.prune_features(dTrain, dTest)
            #     prunedDataTrain.append(dTrainPruned)
            #     prunedDataTest.append(dTestPruned)
            #
            # print("it took", (time.time() - startPruneTime) / 60.0, "minutes")
            print('learning bag of lesions...')

            startBol = time.time()
            allTrainData, mixture_models = learn_bol(train_patients, trainDataVectors, len(mri_train), results_dir, foldNum)
            elapsedBol = time.time() - startBol
            print(str(elapsedBol / 60), 'minutes to learn BoL.')

            print('transforming test data to bag of lesions representation...')
            allTestData = project_to_bol(mri_test, dataVectorsTest, mixture_models)

            print('train BoL shape:', allTrainData.shape)
            print('test BoL shape:', allTestData.shape)

            trainingPatientsByTreatment, testingPatientsByTreatment, trainingData, testingData, trainCounts, testCounts = separatePatientsByTreatment(mri_train, mri_test, allTrainData, allTestData, trainCounts, testCounts)

            # feature selection
            featuresToRemove, c = None, None
            for treatment in treatments:
                if True:
                # try:
                    trainData, testData = trainingData[treatment], testingData[treatment]
                    trainDataCopy, testDataCopy = trainData, testData
                    trainOutcomes, testOutcomes = load_data.get_outcomes(trainingPatientsByTreatment[treatment]), load_data.get_outcomes(
                        testingPatientsByTreatment[treatment])

                    remove_worst_features = False
                    if remove_worst_features:
                        if treatment == "Placebo":
                            print('selecting features...')
                            bestTrainData, bestTestData, featuresToRemove = bol_classifiers.randomForestFeatureSelection(
                                trainDataCopy, testDataCopy, trainOutcomes, testOutcomes, 12)
                        else:
                            bestTrainData, bestTestData = removeWorstFeatures(trainDataCopy, testDataCopy, featuresToRemove)
                    else:
                        bestTrainData = trainDataCopy
                        bestTestData = testDataCopy

                    if treatment == "Placebo":
                        (bestFeatureScore, bestFeaturePredictions, placebo_rf), (probScore, probPredicted), (
                        correct, total) = bol_classifiers.random_forest(bestTrainData, bestTestData, trainOutcomes,
                                                                            testOutcomes, mri_test, mixture_models, results_dir)

                        activity_truth[treatment].append(testOutcomes)
                        activity_posterior[treatment].append(probPredicted)

                        svm_linear_posterior, svm_rbf_posterior, chi2svm_posterior = bol_classifiers.svms(bestTrainData, bestTestData, trainOutcomes)
                        knn_euclid_posterior, knn_maha_posterior = bol_classifiers.knn(bestTrainData, trainOutcomes, bestTestData)

                        chi2_svm_posterior[treatment].append(chi2svm_posterior)
                        rbf_svm_posterior[treatment].append(svm_rbf_posterior)
                        linear_svm_posterior[treatment].append(svm_linear_posterior)

                        euclidean_knn_posterior[treatment].append(knn_euclid_posterior)
                        mahalanobis_knn_posterior[treatment].append(knn_maha_posterior)

                        naive_bayes_posterior[treatment].append([])   # FIX IT

                    # drugged patients
                    else:
                        # project onto untreated MS model (don't train)
                        (bestPreTrainedFeatureScore, bestPreTrainedFeaturePredictions, meh), (
                        pretrainedProbScore, pretrainedProbPredicted), (correct, total) = bol_classifiers.random_forest(
                            bestTrainData, bestTestData, trainOutcomes, testOutcomes, mri_test, mixture_models, results_dir, placebo_rf)

                        # new model on drugged patients
                        (bestFeatureScore, bestFeaturePredictions, drug_rf), (probScore, probDrugPredicted), (
                        correct, total) = bol_classifiers.random_forest(bestTrainData, bestTestData, trainOutcomes,
                                                                            testOutcomes, mri_test, mixture_models, results_dir)

                        svm_linear_posterior, svm_rbf_posterior, chi2svm_posterior = bol_classifiers.svms(bestTrainData, bestTestData, trainOutcomes)
                        knn_euclid_posterior, knn_maha_posterior = bol_classifiers.knn(bestTrainData, trainOutcomes, bestTestData)

                        activity_truth[treatment].append(testOutcomes)
                        activity_posterior[treatment].append(np.asarray(probDrugPredicted))
                        untreated_posterior[treatment].append(np.asarray(pretrainedProbPredicted))

                        chi2_svm_posterior[treatment].append(chi2svm_posterior)
                        rbf_svm_posterior[treatment].append(svm_rbf_posterior)
                        linear_svm_posterior[treatment].append(svm_linear_posterior)
                        euclidean_knn_posterior[treatment].append(knn_euclid_posterior)
                        mahalanobis_knn_posterior[treatment].append(knn_maha_posterior)

                        certainNumber[treatment] += total
                        certainCorrect[treatment] += correct

                        right, wrong, r1_score, r2_score, r3_score, r4_score, responders_this_fold = showWhereTreatmentHelped(
                            pretrainedProbPredicted, probDrugPredicted, bestTrainData, bestTestData, trainOutcomes,
                            testOutcomes, trainingPatientsByTreatment[treatment],
                            testingPatientsByTreatment[treatment], results_dir)

                        for responder in responders_this_fold:
                            responder_writer.writerow([responder['uid'], responder['treatment'], responder['t2_lesions'], responder['P(A=1|BoL, untr)'], responder['P(A=0|BoL, tr)']])

                        respondersRight[treatment] += right
                        respondersWrong[treatment] += wrong

                        print('responders right', respondersRight, 'responders wrong', respondersWrong)

                        (responderScore, responderProbs), responderHighProbScore = bol_classifiers.identify_responders(
                            bestTrainData, bestTestData, trainOutcomes, testOutcomes, trainCounts[treatment],
                            testCounts[treatment], drug_rf, placebo_rf)

                    certainNumberPre[treatment] += total
                    certainCorrectPre[treatment] += correct

                    # for scoreMet in scoringMetrics + ['sensitivity', 'specificity']:
                    #     featureScores[treatment][scoreMet].append(bestFeatureScore[scoreMet])
                    #
                    #     # bad classifiers
                    #     # bestKnnEuclideanScores[treatment][scoreMet].append(bestKnnEuclideanScoreVals[scoreMet])
                    #     # bestKnnMahalanobisScores[treatment][scoreMet].append(bestKnnMahalanobisScoreVals[scoreMet])
                    #     # bestChi2Scores[treatment][scoreMet].append(bestChi2Score[scoreMet])
                    #     # bestChi2svmScores[treatment][scoreMet].append(bestChi2svmscore[scoreMet])
                    #     # bestFeatureScores[treatment][scoreMet].append(bestFeatureScore[scoreMet])
                    #     # bestSvmLinScores[treatment][scoreMet].append(bestSvmLinScore[scoreMet])
                    #     # bestSvmRadScores[treatment][scoreMet].append(bestSvmRadScore[scoreMet])
                    #     # countingScores[treatment][scoreMet].append(countingScore[scoreMet])
                    #     # probScores[treatment][scoreMet].append(probScore[scoreMet])
                    #     # allProbScores[treatment][scoreMet].append(probScore[scoreMet])
                    #
                    #     if treatment != "Placebo":
                    #         preTrainedFeatureScores[treatment][scoreMet].append(bestPreTrainedFeatureScore[scoreMet])
                    #         responderScores[treatment][scoreMet].append(responderScore[scoreMet])
                    #         responderHighProbScores[treatment][scoreMet].append(responderHighProbScore[scoreMet])
                    #         # countScores[treatment][scoreMet].append(count_score[scoreMet])
                    #
                    #         r1[treatment][scoreMet].append(r1_score[scoreMet])
                    #         r2[treatment][scoreMet].append(r2_score[scoreMet])
                    #         r3[treatment][scoreMet].append(r3_score[scoreMet])
                    #         r4[treatment][scoreMet].append(r4_score[scoreMet])

                # except Exception as e:
                #     print('ERROR:', e)
                #     failedFolds += 1
                #     scoreThisFold = False

                # if scoreThisFold:
                #     for treatment in treatments:
                #         if treatment == "Placebo":
                #             bestScoring = []
                #             # bestScoring.append((bestKnnEuclideanScores[treatment], "NN-Euclidean"))
                #             # bestScoring.append((bestKnnMahalanobisScores[treatment], "NN-Mahalanobis"))
                #             # bestScoring.append((bestChi2Scores[treatment], "NN-$\chi^2$"))
                #             #
                #             # bestScoring.append((bestSvmLinScores[treatment], "SVM-Linear"))
                #             # bestScoring.append((bestSvmRadScores[treatment], "SVM-RBF"))
                #             # bestScoring.append((bestChi2svmScores[treatment], "SVM-$\chi^2$"))
                #
                #             bestScoring.append((bestFeatureScores[treatment], "Random Forest"))
                #             # bestScoring.append((countingScores[treatment], "Naive Bayes (Lesion Counts)"))
                #
                #             plotScores(bestScoring, 'Predicting Future MS Lesion Activity', results_dir)
                #         else:
                #             predictionScores = []
                #             predictionScores.append(bestFeatureScores[treatment])
                #
                #             plotScores(bestScoring, 'Predicting Future MS Lesion Activity (' + treatment + ')', results_dir)
                #
                #
                #         # if treatment == "Placebo":
                #         #     bestScoring = []
                #         #
                #         #     bestScoring.append((featureScores[treatment], "Random Forest (all lesions)"))
                #         #     bestScoring.append((allProbScores[treatment], "Random Forest (all lesions, certain)"))
                #         #
                #         #     bestScoring.append((bestFeatureScores[treatment], "Random Forest (best lesions)"))
                #         #     bestScoring.append((probScores[treatment], "Random Forest (best lesions, certain)"))
                #
                #     for treatment in treatments:
                #         if treatment == "Avonex":
                #             plotScores([(r1[treatment], 'Responders'), (r2[treatment], 'Responders (certain GT)'),
                #                         (r3[treatment], 'Responders (certain prediction)'),
                #                         (r4[treatment], 'Responders (all certain)')], "Avonex Responder Prediction", results_dir)
                #         elif treatment == "Laquinimod":
                #             plotScores([(r1[treatment], 'Responders'), (r2[treatment], 'Responders (certain GT)'),
                #                         (r3[treatment], 'Responders (certain prediction)'),
                #                         (r4[treatment], 'Responders (all certain)')], "Laquinimod Responder Prediction", results_dir)
                #
                #     bestScoring = []
                #
                #     for treatment in treatments:
                #         if treatment == "Placebo":
                #             bestScoring.append((bestFeatureScores[treatment], 'Untreated ($\\alpha=0.5$)'))
                #             bestScoring.append((probScores[treatment], 'Untreated ($\\alpha=0.8$)'))
                #
                #         if treatment == "Avonex":
                #             bestScoring.append((preTrainedFeatureScores[treatment], 'Untreated Predictor on Drug A'))
                #             bestScoring.append((bestFeatureScores[treatment], 'Drug A ($\\alpha=0.5$)'))
                #             bestScoring.append((probScores[treatment], 'Drug A ($\\alpha=0.8$)'))
                #
                #         if treatment == "Laquinimod":
                #             bestScoring.append((preTrainedFeatureScores[treatment], 'Untreated Predictor on Drug B'))
                #             bestScoring.append((bestFeatureScores[treatment], 'Drug B ($\\alpha=0.5$)'))
                #             bestScoring.append((probScores[treatment], 'Drug B ($\\alpha=0.8$)'))
                #
                #     plotScores(bestScoring, "Activity Prediction", results_dir)


        best_p_a, best_p_d = responder_roc(activity_truth, activity_posterior, untreated_posterior, results_dir)

        activity_posteriors = [activity_posterior, euclidean_knn_posterior, linear_svm_posterior, chi2_svm_posterior, rbf_svm_posterior]
        classifier_names = ['Random Forest', '1-NN (Euclidean)', 'SVM (linear)', 'SVM ($\\chi^2$)', 'SVM (RBF)']
        colours = ['darkred', 'indianred', 'lightsalmon', 'darkorange', 'goldenrod', 'tan']

        for treatment in treatments:
            # print('GT:', np.asarray(activity_truth[treatment][0]).shape, np.asarray(activity_truth[treatment][1]).shape)
            # print('Predictions:', np.asarray(activity_posterior[treatment][0]).shape, np.asarray(activity_posterior[treatment][1]).shape)

            plt.figure(figsize=(8,8))
            lw = 2
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

            y_true = np.concatenate(tuple(activity_truth[treatment]), axis=0)
            for p, (probabilities, colour) in enumerate(zip(activity_posteriors, colours)):
                y_prob = np.concatenate(tuple(probabilities[treatment]), axis=0)

                roc_auc = roc_auc_score(y_true, y_prob[:, 1], 'weighted')
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                plt.plot(fpr, tpr, color=colour, lw=lw, label=classifier_names[p] + ' ROC (area = %0.2f)' % roc_auc)

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=20)
            plt.ylabel('True Positive Rate', fontsize=20)
            # plt.title('Receiver operating characteristic example', fontsize=24)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True, fontsize=20)
            plt.savefig(results_dir + 'activity_prediction_' + treatment + '_roc.png', bbox_inches='tight')

        # print("FAILED FOLDS:", failedFolds)

        print('certain correct pretrained', certainCorrectPre)
        print('certain total pretrained', certainNumberPre)

        print('certain correct', certainCorrect)
        print('certain total', certainNumber)

        end = time.time()
        elapsed = end - start
        print(str(elapsed / 60), 'minutes elapsed.')

    return experiment_number

if __name__ == "__main__":
    experiment_number = predict_responders()
    print('This experiment was brought to you by the number:', )