import os, pickle, time, csv
from collections import defaultdict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.mixture import GMM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve

import umap
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

import load_data
import bol_classifiers
from analyze_lesions import learn_bol, project_to_bol, separatePatientsByTreatment, choose_clusters

from mri import mri

treatments = ['Placebo', 'Laquinimod', 'Avonex']
treatment_labels = ['Placebo', 'Drug A', 'Drug B']

classifier_names = ['Random Forest', 'SVM (Linear)', 'SVM (RBF)', '1-NN ($\\chi^2$)', '1-NN (Mahalanobis)']

modalities = ['t1p', 't2w', 'pdw', 'flr']
tissues = ['csf', 'wm', 'gm', 'pv', 'lesion']

feats = ["Context", "RIFT", "LBP", "Intensity"]

scoringMetrics = ['TP', 'FP', 'TN', 'FN']

metrics = ['newT2']

datadir = '/data1/users/adoyle/MS-LAQ/MS-LAQ-302-STX/'
responder_filename = 'Bravo_responders.csv'

mri_list_location = datadir + 'mri_list.pkl'

n_folds = 25

def responder_roc(all_test_patients, activity_truth, activity_posterior, untreated_posterior, results_dir):

    print('Untreated posteriors:', untreated_posterior['Placebo'])
    print('Activity posteriors:', activity_posterior)
    print('Activity truth:', activity_truth)

    # print('Untreated posteriors shape:', untreated_posterior['Placebo'].shape)
    # print('Activity posteriors:', activity_posterior['Avonex'].shape)
    # print('Activity truth:', activity_truth['Avonex'].shape)

    with open(results_dir + 'responders.csv', 'w') as csvfile:
        responder_writer = csv.writer(csvfile)
        responder_writer.writerow(
            ['Subject ID', 'Treatment', '# T2 Lesions', 'P(A=1|BoL, untr)', 'P(A=0|BoL, tr)', 'Responder'])
        mri_list = defaultdict(list)
        for treatment in treatments:
            for sublists in all_test_patients[treatment]:
                for mri in sublists:
                    mri_list[treatment].append(mri)

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

                a_range = np.linspace(0, 1, n_folds, endpoint=False)
                d_range = np.linspace(0, 1, n_folds, endpoint=False)

                fig2 = plt.figure(1, figsize=(10, 10))
                ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((3, 1), (2, 0))

                ax1.plot(a_range, a_range, "k:", label="Perfectly calibrated")

                for n_a, p_a in enumerate(a_range):
                    try:
                        a_true_inferred = np.zeros(a_prob.shape)
                        a_true_inferred[a_prob > p_a] = 1

                        # print('A untreated predictions:', a_true_inferred)
                        fraction_of_positives, mean_predicted_value = calibration_curve(a_true_inferred, a_prob, n_bins=10)

                        score = brier_score_loss(a_true_inferred, a_prob)
                        p_a_brier.append(score)

                        if n_a%5 == 0:
                            ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (str(p_a), score))

                    except:
                        p_a_brier.append(1)

                    try:
                        # tn, tp, _ = roc_curve(a_true_inferred, a_prob)
                        auc_weighted = roc_auc_score(a_true_inferred, a_prob, 'weighted')
                        auc_macro = roc_auc_score(a_true_inferred, a_prob, 'macro')
                        auc_micro = roc_auc_score(a_true_inferred, a_prob, 'micro')
                        auc_samples = roc_auc_score(a_true_inferred, a_prob, 'samples')

                        # print('AUCs (weighted, macro, micro, samples):', auc_weighted, auc_macro, auc_micro, auc_samples)

                        p_a_auc.append(auc_macro)
                    except:
                        print('AUC undefined for:', p_a)
                        p_a_auc.append(0)


                ax2.hist(a_prob, range=(0, 1), bins=20, label='P(A|BoL, untr)', histtype="step", lw=2)

                ax1.set_ylabel("Fraction of positives")
                ax1.set_ylim([-0.05, 1.05])
                ax1.legend(loc="lower right", shadow=True)
                ax1.set_title('Calibration plots  (reliability curve)')

                ax2.set_xlabel("Mean predicted value")
                ax2.set_ylabel("Count")
                ax2.legend(loc="upper center", ncol=2, shadow=True)

                plt.tight_layout()
                plt.savefig(results_dir + treatment + '_calibration_curve.png')

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
                r_predicted[a_prob > best_p_a] = 1            # and inactive on the drug
                r_predicted[d_predicted < best_p_d] = 0

                roc_auc = roc_auc_score(r_true, r_predicted, 'weighted')
                fpr, tpr, _ = roc_curve(r_true, r_predicted)

                plt.figure(0)
                lw = 2
                if 'Laquinimod' in treatment:
                    ax.plot(fpr, tpr, color='darkorange', lw=lw, label=treatment + ' ROC (AUC = %0.2f)' % roc_auc)
                    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                else:
                    ax.plot(fpr, tpr, color='darkred', lw=lw, label=treatment + ' ROC (AUC = %0.2f)' % roc_auc)

                # plt.title('Receiver operating characteristic example', fontsize=24)

                print(treatment + ' optimal thresholds (activity, drug_activity): ', best_p_a, best_p_d)

                for i in range(len(mri_list[treatment])):
                    scan = mri_list[treatment][i]
                    treatment = scan.treatment
                    t2_les = str(scan.newT2)
                    p_a_untr = str(a_prob[i])
                    p_a_tr = str(d_prob[i])
                    respond = str(r_predicted[i])

                    responder_writer.writerow([scan.uid, treatment, t2_les, p_a_untr, p_a_tr, respond])

        plt.figure(0)
        ax.set_xlabel('False Positive Rate', fontsize=20)
        ax.set_ylabel('True Positive Rate', fontsize=20)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True, fontsize=20)
        plt.savefig(results_dir + 'responder_' + 'p_a_'+ str(best_p_a) + '_p_d_' + str(best_p_d) + '_roc.png', bbox_inches='tight')

    return best_p_a, best_p_d

def cluster_stability(bol_mixtures, random_forests, results_dir):
    n_folds = len(bol_mixtures)

    n_lesion_types_all_folds = 0

    feature_dims = bol_mixtures[0].means_.shape[1]

    for k in range(n_folds):
        n_lesion_types_all_folds += len(bol_mixtures[k].weights_)

    all_lesion_types = np.zeros((n_lesion_types_all_folds, feature_dims))
    all_type_weights = np.zeros((n_lesion_types_all_folds))

    all_lesion_importances = np.zeros((n_folds, random_forests['Placebo'][0].feature_importances_.shape[0]))

    n_components = []

    lesion_type_means = np.zeros((n_folds, feature_dims))

    idx = 0
    for fold, (mixture_model, rf_placebo, rf_avonex, rf_laquinimod) in enumerate(zip(bol_mixtures, random_forests['Placebo'], random_forests['Avonex'], random_forests['Laquinimod'])):
        n_components.append(len(mixture_model.weights_))

        all_lesion_importances[fold, :] += rf_placebo.feature_importances_
        # all_lesion_importances[fold, :] += rf_avonex.feature_importances_
        # all_lesion_importances[fold, :] += rf_laquinimod.feature_importances_

        for lesion_type_centre, type_weight in zip(mixture_model.means_, mixture_model.weights_):
            all_lesion_types[idx, :] = lesion_type_centre
            all_type_weights[idx] = type_weight

            idx += 1

    all_type_weights *= (10/np.max(all_type_weights))

    n_lesion_types_first_fold = len(bol_mixtures[0].weights_)
    lesion_type_labels = np.arange(n_lesion_types_first_fold)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(all_lesion_types[0:n_lesion_types_first_fold, :], lesion_type_labels)

    corresponding_lesion_types = knn.predict(all_lesion_types)
    print('corresponding lesion types:', corresponding_lesion_types.shape)

    embedded_umap = umap.UMAP(metric='mahalanobis').fit_transform(all_lesion_types)
    embedded_tsne = TSNE(random_state=42, metric='mahalanobis').fit_transform(all_lesion_types)

    print('t-sne embedded shape:', embedded_tsne.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6), dpi=500)

    cmap = mpl.cm.get_cmap('rainbow')

    for label in lesion_type_labels:
        for predicted_label, (x_tsne, y_tsne), (x_umap, y_umap), weight in zip(corresponding_lesion_types, embedded_tsne, embedded_umap, all_type_weights):
            if label == predicted_label:
                ax1.scatter(x_tsne, y_tsne, s=40**weight, color=cmap((label+1)/len(lesion_type_labels)))
                ax2.scatter(x_umap, y_umap, s=40**weight, color=cmap((label+1)/len(lesion_type_labels)))

    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(results_dir + 'embedding_of_lesion_types.png', dpi=600)

    # boxplot for lesion-type importance across folds

    corresponding_lesion_type_importance = []
    for n in range(n_lesion_types_first_fold):
        corresponding_lesion_type_importance.append([])

    for fold, type_importances in enumerate(all_lesion_importances):
        #types don't correspond yet
        fold_type_labels = corresponding_lesion_types[(fold)*n_lesion_types_first_fold:(fold+1)*n_lesion_types_first_fold, :]

        for type_number in fold_type_labels:
            corresponding_lesion_type_importance[type_number].append(type_importances[type_number])

    fig, axes = plt.subplots(1, 1, figsize=(6, 6), dpi=600)

    axes[0].boxplot(corresponding_lesion_type_importance)
    axes[0].set_ylabel('Lesion-type importance', fontsize=20)
    axes[0].set_xlabel('Lesion-type', fontsize=20)

    plt.tight_layout()
    plt.savefig(results_dir + 'corresponding_lesion_importance.png', bbox_inches='tight')


    fig, axes = plt.subplots(1, 3)

    data = [n_components]
    print('lesion-types:', data)
    axes[0].boxplot(data)
    axes[0].set_ylabel('Number of lesion-types', fontsize=20)

    importance = np.zeros((n_folds, np.max(n_components)))

    for fold, mixture_models in enumerate(bol_mixtures):
        importance_start_idx = 0

        rfs = random_forests['Placebo']
        lesion_importance = rfs[fold].feature_importances_

        sorted_indices = np.argsort(mixture_models.weights_)

        for c, cluster_idx in enumerate(sorted_indices):
            lesion_type_means[fold, :] = mixture_models.means_[cluster_idx, :]
            importance[fold, c] = lesion_importance[importance_start_idx+c]

        importance_start_idx += len(sorted_indices)


    dim_mean = np.mean(lesion_type_means, axis=0)
    dim_var = np.var(lesion_type_means, axis=0)

    print('cluster centre means:', dim_mean.shape)
    print('cluster centre variances:', dim_var.shape)

    diffs = []

    for fold, lesion_type_centre in enumerate(lesion_type_means):
        print('lesion type centre:', lesion_type_centre.size)

        diff = np.subtract(lesion_type_centre, dim_mean)
        diff_normalized = np.divide(diff, dim_var)

        diffs.append(diff_normalized)

    data2 = [diffs]
    print(data2)

    axes[1].boxplot(data2)
    axes[1].set_xlabel('Lesion size', fontsize=20)
    axes[1].set_ylabel('Diff. from mean', fontsize=20)


    data3 = [importance[:, 0], importance[:, 1], importance[:, 2], importance[:,3]]
    axes[2].boxplot(data3)
    axes[2].set_xlabel('Lesion size', fontsize=20)
    axes[2].set_ylabel('P(A|BoL) Importance', fontsize=20)

    plt.tight_layout()
    plt.savefig(results_dir + 'cluster_numbers_lesion_centres.png', bbox_inches='tight')


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

    features = load_data.loadAllData(mri_list)
    # n_lesion_types = choose_clusters(features, results_dir)

    n_lesion_types = 12

    mri_list, without_clinical = load_data.loadClinical(mri_list)

    print('We have ' + str(len(mri_list)) + ' patients who finished the study and ' + str(len(without_clinical)) + ' who did not')
    outcomes = load_data.get_outcomes(mri_list)

    mri_list = load_data.load_responders(datadir + responder_filename, mri_list)

    patient_results = {}
    for scan in mri_list:
        patient_results[scan.uid] = {}

    kf = StratifiedKFold(n_folds, shuffle=True, random_state=42)

    bol_mixture_models = []
    random_forests = defaultdict(list)

    all_test_patients, activity_posterior, activity_truth, untreated_posterior = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    euclidean_knn_posterior, mahalanobis_knn_posterior, chi2_svm_posterior, rbf_svm_posterior, linear_svm_posterior, naive_bayes_posterior =  defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

    # initialization of result structures complete
    # start learning BoL, predicting activity
    for foldNum, (train_index, test_index) in enumerate(kf.split(range(len(mri_list)), outcomes)):
        print(foldNum+1, '/', kf.get_n_splits())

        mri_train, mri_test = np.asarray(mri_list)[train_index], np.asarray(mri_list)[test_index]

        # incorporate patients with no clinical data
        train_patients = []
        for scan in mri_train:
            train_patients.append(scan)
        for scan in without_clinical:
            train_patients.append(scan)

        # print('loading feature data...')
        # startLoad = time.time()
        raw_train_data = load_data.loadAllData(train_patients)
        raw_test_data = load_data.loadAllData(mri_test)

        print('learning bag of lesions...')

        startBol = time.time()
        bol_train_data, mixture_model = learn_bol(train_patients, raw_train_data, n_lesion_types, len(mri_train), results_dir, foldNum)

        bol_mixture_models.append(mixture_model)

        elapsedBol = time.time() - startBol
        print(str(elapsedBol / 60), 'minutes to learn BoL.')

        print('transforming test data to bag of lesions representation...')
        bol_test_data = project_to_bol(mri_test, raw_test_data, mixture_model)

        print('train BoL shape:', bol_train_data.shape)
        print('test BoL shape:', bol_test_data.shape)

        trainingPatientsByTreatment, testingPatientsByTreatment, trainingData, testingData = separatePatientsByTreatment(mri_train, mri_test, bol_train_data, bol_test_data)

        # feature selection
        for treatment in treatments:
            train_data, test_data = trainingData[treatment], testingData[treatment]
            train_outcomes, test_outcomes = load_data.get_outcomes(trainingPatientsByTreatment[treatment]), load_data.get_outcomes(
                testingPatientsByTreatment[treatment])

            all_test_patients[treatment].append(testingPatientsByTreatment[treatment])

            if treatment == "Placebo":
                (bestFeaturePredictions, placebo_rf, probPredicted) = bol_classifiers.random_forest(train_data, test_data, train_outcomes)

                random_forests[treatment].append(placebo_rf)
                activity_truth[treatment].append(test_outcomes)
                activity_posterior[treatment].append(probPredicted)

                svm_linear_posterior, svm_rbf_posterior, chi2svm_posterior = bol_classifiers.svms(train_data, test_data, train_outcomes)
                knn_euclid_posterior, knn_maha_posterior = bol_classifiers.knn(train_data, train_outcomes, test_data)

                chi2_svm_posterior[treatment].append(chi2svm_posterior)
                rbf_svm_posterior[treatment].append(svm_rbf_posterior)
                linear_svm_posterior[treatment].append(svm_linear_posterior)

                euclidean_knn_posterior[treatment].append(knn_euclid_posterior)
                mahalanobis_knn_posterior[treatment].append(knn_maha_posterior)

                naive_bayes_posterior[treatment].append([])   # FIX IT

            # drugged patients
            else:
                # project onto untreated MS model (don't train)
                (bestPreTrainedFeaturePredictions, meh, pretrainedProbPredicted) = bol_classifiers.random_forest(
                    train_data, test_data, train_outcomes, placebo_rf)

                # new model on drugged patients
                (bestFeaturePredictions, drug_rf, probDrugPredicted) = bol_classifiers.random_forest(train_data, test_data, train_outcomes)

                random_forests[treatment].append(drug_rf)
                svm_linear_posterior, svm_rbf_posterior, chi2svm_posterior = bol_classifiers.svms(train_data, test_data, train_outcomes)
                knn_euclid_posterior, knn_maha_posterior = bol_classifiers.knn(train_data, train_outcomes, test_data)

                activity_truth[treatment].append(test_outcomes)
                activity_posterior[treatment].append(np.asarray(probDrugPredicted))
                untreated_posterior[treatment].append(np.asarray(pretrainedProbPredicted))

                chi2_svm_posterior[treatment].append(chi2svm_posterior)
                rbf_svm_posterior[treatment].append(svm_rbf_posterior)
                linear_svm_posterior[treatment].append(svm_linear_posterior)
                euclidean_knn_posterior[treatment].append(knn_euclid_posterior)
                mahalanobis_knn_posterior[treatment].append(knn_maha_posterior)

                # right, wrong, r1_score, r2_score, r3_score, r4_score, responders_this_fold = showWhereTreatmentHelped(
                #     pretrainedProbPredicted, probDrugPredicted, train_data, test_data, train_outcomes,
                #     test_outcomes, trainingPatientsByTreatment[treatment],
                #     testingPatientsByTreatment[treatment], results_dir)
                #
                # (responderScore, responderProbs), responderHighProbScore = bol_classifiers.identify_responders(
                #     train_data, test_data, train_outcomes, test_outcomes, trainCounts[treatment],
                #     testCounts[treatment], drug_rf, placebo_rf)


    best_p_a, best_p_d = responder_roc(all_test_patients, activity_truth, activity_posterior, untreated_posterior, results_dir)

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

    end = time.time()
    elapsed = end - start

    cluster_stability(bol_mixture_models, random_forests, results_dir)
    print(str(elapsed / 60), 'minutes elapsed.')

    return experiment_number

if __name__ == "__main__":
    experiment_number = predict_responders()
    print('This experiment was brought to you by the number:', )