import os, pickle, time, csv
from collections import defaultdict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse

from sklearn.mixture import GMM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve

import umap
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric

import load_data
import bol_classifiers
from analyze_lesions import learn_bol, project_to_bol, separatePatientsByTreatment, choose_clusters
from feature_extraction import write_features

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

def responder_roc(all_test_patients, activity_truth, activity_posterior, untreated_posterior, n_folds, results_dir):

    # print('Untreated posteriors:', untreated_posterior['Placebo'])
    # print('Activity posteriors:', activity_posterior)
    # print('Activity truth:', activity_truth)

    # print('Untreated posteriors shape:', untreated_posterior['Placebo'].shape)
    # print('Activity posteriors:', activity_posterior['Avonex'].shape)
    # print('Activity truth:', activity_truth['Avonex'].shape)

    with open(results_dir + 'responders.csv', 'w') as csvfile:
        responder_writer = csv.writer(csvfile)
        responder_writer.writerow(
            ['Subject ID', 'Treatment', '# T2 Lesions', 'P(A=1|BoL, untr)', 'P(A=1|BoL, tr)', 'Responder'])
        mri_list = defaultdict(list)
        for treatment in treatments:
            for sublists in all_test_patients[treatment]:
                for mri in sublists:
                    mri_list[treatment].append(mri)

        fig1 = plt.figure(0, dpi=500) # roc
        fig2 = plt.figure(1, dpi=500) # predictions distribution

        ax = fig1.add_subplot(1, 1, 1)
        ax1 = fig2.add_subplot(1, 1, 1)

        for treatment, treat in zip(treatments, treatment_labels):
            p_a_auc, p_d_distance, p_d_harmonic_mean, p_d_anti_harmonic_mean, p_a_brier = [], [], [], [], []

            if 'Placebo' not in treatment:
                a_prob = np.concatenate(tuple(untreated_posterior[treatment]), axis=0) # must use predictions for untreated
                a_prob = a_prob[:, 1]                                                  # to infer what would have happened if untreated

                # print('Untreated predictions (' + treatment + '):', a_prob)

                d_true = np.concatenate(tuple(activity_truth[treatment]), axis=0)
                d_prob = np.concatenate(tuple(activity_posterior[treatment]), axis=0)

                d_prob = d_prob[:, 1]                          # just consider the P(A=1|BoL, treatment)

                a_range = np.linspace(0, 1, n_folds, endpoint=False)
                d_range = np.linspace(0, 1, n_folds, endpoint=False)

                for n_a, p_a in enumerate(a_range):
                    try:
                        a_true_inferred = np.zeros(a_prob.shape)
                        a_true_inferred[a_prob > p_a] = 1

                        # print('A untreated predictions:', a_true_inferred)
                        fraction_of_positives, mean_predicted_value = calibration_curve(a_true_inferred, a_prob, n_bins=10)

                        score = brier_score_loss(a_true_inferred, a_prob)
                        p_a_brier.append(score)

                        # if n_a%5 == 0:
                        #     ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (str(p_a), score))

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


                ax1.hist(a_prob, range=(0, 1), bins=20, label='P(A=1|BoL, untr) for ' + treat + ' subjs', histtype="step", lw=2)
                ax1.hist(d_prob, range=(0, 1), bins=20, label='P(A=1|BoL, ' + treat + ')', histtype='step', lw=2)

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

                        epsilon = 1e-6
                        sens = tp/(tp + fn + epsilon)
                        spec = tn/(tn + fp + epsilon)

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
                    ax.plot(fpr, tpr, color='darkorange', lw=lw, label=treat + ' ROC (AUC = %0.2f)' % roc_auc)
                    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                else:
                    ax.plot(fpr, tpr, color='darkred', lw=lw, label=treat + ' ROC (AUC = %0.2f)' % roc_auc)

                # plt.title('Receiver operating characteristic example', fontsize=24)

                print(treatment + ' optimal thresholds (activity, drug_activity): ', best_p_a, best_p_d)

                # untreated_threshold, treated_threshold = 0.8, 0.2

                untreated_thresholds = np.linspace(0, 1)
                treated_thresholds = np.linspace(0, 1)

                responder_results = np.zeros((untreated_thresholds.shape[0], treated_thresholds.shape[0], 4))

                for i, untreated_threshold in enumerate(untreated_thresholds):
                    for j, treated_threshold in enumerate(treated_thresholds):
                        responder_list, actual_outcome_list = [], []

                        for p_activity_untreated, p_activity_treated, activity in zip(a_prob, d_prob, d_true):
                            # print('P(A=1|BoL, untr), P(A=1|BoL, tr), A', p_activity_untreated, p_activity_treated, activity)
                            if p_activity_untreated > untreated_threshold and p_activity_treated <= treated_threshold:
                                responder_list.append(1)
                                actual_outcome_list.append(activity)
                            # elif p_activity_untreated < untreated_threshold:
                            #     responder_list.append(0)
                            #     actual_outcome_list.append(activity)
                            elif p_activity_untreated > untreated_threshold and p_activity_treated >= treated_threshold:
                                responder_list.append(0)
                                actual_outcome_list.append(activity)

                        if len(responder_list) > 0:
                            tn, fp, fn, tp = confusion_matrix(np.asarray(responder_list), np.asarray(actual_outcome_list), labels=[0, 1]).ravel()

                            epsilon = 1e-6

                            sens = tp/(tp + fn + epsilon)
                            spec = tn/(tn + fp + epsilon)

                            responder_results[i, j, 0] = sens
                            responder_results[i, j, 1] = spec
                            responder_results[i, j, 2] = 2*sens*spec / (sens + spec + epsilon) # harmonic mean!
                            responder_results[i, j, 3] = len(responder_list)

                            # print(untreated_threshold, treated_threshold, sens, spec)
                X, Y = np.meshgrid(untreated_thresholds, treated_thresholds)
                z = responder_results[:, :, 2]

                fig = plt.figure(2, dpi=500)
                ax_thresholds = plt.axes(projection='3d')
                surf = ax_thresholds.plot_surface(X, Y, z, vmin=np.nanmin(z), vmax=np.nanmax(z), rstride=1, cstride=1, cmap='Spectral_r', edgecolor='none')
                ax_thresholds.set_xlabel('P(A=1|BoL, untr)\nthreshold')
                ax_thresholds.set_ylabel('P(A=0|BoL, ' + treat + ')\nthreshold')
                ax_thresholds.set_zlabel('Sens/Spec\n(harmonic mean)')

                ax_thresholds.invert_xaxis()

                fig.colorbar(surf, shrink=0.4, aspect=4)
                plt.savefig(results_dir + treatment + '_thresholds.png')

                flat_index = np.argmax(responder_results[:, :, 2])
                unflat_indices = np.unravel_index(flat_index, (responder_results.shape[0], responder_results.shape[1]))

                best_untreated_threshold = untreated_thresholds[unflat_indices[0]]
                best_treated_threshold = treated_thresholds[unflat_indices[1]]

                for i in range(len(mri_list[treatment])):
                    scan = mri_list[treatment][i]
                    treatment = scan.treatment
                    t2_les = str(scan.newT2)
                    p_a_untr = str(a_prob[i])
                    p_a_tr = str(d_prob[i])
                    respond = str(r_predicted[i])

                    responder_writer.writerow([scan.uid, treatment, t2_les, p_a_untr, p_a_tr, respond])

        plt.figure(1)
        ax1.set_xlabel("Mean predicted value", fontsize=24)
        ax1.set_ylabel("Count", fontsize=24)
        ax1.legend(loc='upper left', shadow=True, fancybox=True, fontsize=20)
        plt.savefig(results_dir + 'prediction_distribution.png')

        plt.figure(0)
        ax.set_xlabel('False Positive Rate', fontsize=20)
        ax.set_ylabel('True Positive Rate', fontsize=20)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True, fontsize=20)
        plt.savefig(results_dir + 'responder_' + 'p_a_'+ str(best_p_a) + '_p_d_' + str(best_p_d) + '_roc.png', bbox_inches='tight')

    return best_p_a, best_p_d


def cluster_stability(bol_mixtures, random_forests, lime_importances, results_dir):
    n_folds = len(bol_mixtures)

    n_lesion_types_all_folds = 0

    feature_dims = bol_mixtures[0].means_.shape[1]

    for k in range(n_folds):
        n_lesion_types_all_folds += len(bol_mixtures[k].weights_)

    all_lesion_types = np.zeros((n_lesion_types_all_folds, feature_dims + 1))
    # all_type_weights = np.zeros((n_lesion_types_all_folds))

    all_lesion_importances = np.zeros((n_folds, random_forests['Placebo'][0].feature_importances_.shape[0]))
    all_lime_importances = np.zeros((n_folds, random_forests['Placebo'][0].feature_importances_.shape[0]))

    n_components = []

    idx = 0
    for fold, (mixture_model, rf_placebo, rf_avonex, rf_laquinimod) in enumerate(zip(bol_mixtures, random_forests['Placebo'], random_forests['Avonex'], random_forests['Laquinimod'])):
        n_components.append(len(mixture_model.weights_))

        all_lesion_importances[fold, :] += rf_placebo.feature_importances_
        all_lesion_importances[fold, :] += rf_avonex.feature_importances_
        all_lesion_importances[fold, :] += rf_laquinimod.feature_importances_

        all_lime_importances[fold, :] += lime_importances['Placebo'][fold]
        all_lime_importances[fold, :] += lime_importances['Avonex'][fold]
        all_lime_importances[fold, :] += lime_importances['Laquinimod'][fold]

        for lesion_type_centre, type_weight in zip(mixture_model.means_, mixture_model.weights_):
            all_lesion_types[idx, :-1] = lesion_type_centre
            all_lesion_types[idx, -1] = type_weight

            idx += 1

    # all_type_weights *= (10/np.max(all_type_weights))

    n_lesion_types_first_fold = len(bol_mixtures[0].weights_)
    lesion_type_labels = np.arange(n_lesion_types_first_fold)

    first_fold_lesion_types = all_lesion_types[0:n_lesion_types_first_fold, :]

    # V = np.cov(all_lesion_types)
    # mahalanobis_distance = DistanceMetric.get_metric('mahalanobis', V=np.cov(V))

    # sort lesion types by greatest cluster separation and then iterate through folds?
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(first_fold_lesion_types, lesion_type_labels)

    corresponding_lesion_types = knn.predict(all_lesion_types)
    print('corresponding lesion types:', corresponding_lesion_types.shape)

    embedded_umap = umap.UMAP(metric='euclidean', random_state=42).fit_transform(all_lesion_types)
    embedded_tsne = TSNE(random_state=42, metric='euclidean').fit_transform(all_lesion_types)

    print('t-sne embedded shape:', embedded_tsne.shape)
    print('umap embedded shape:', embedded_umap.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=500)

    cmap = mpl.cm.get_cmap('rainbow')
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'x', 'D']
    type_markers = []

    for n in range(n_lesion_types_first_fold):
        type_markers.append(markers[np.random.randint(len(markers))])

    for label in lesion_type_labels:
        for predicted_label, (x_tsne, y_tsne), (x_umap, y_umap) in zip(corresponding_lesion_types, embedded_tsne, embedded_umap):
            if label == predicted_label:
            #     ax1.scatter(x_tsne, y_tsne, s=4**weight, color=cmap((label+1)/len(lesion_type_labels)))
            #     ax2.scatter(x_umap, y_umap, s=4**weight, color=cmap((label+1)/len(lesion_type_labels)))

                ax1.scatter(x_tsne, y_tsne, color=cmap((label+1)/len(lesion_type_labels)), marker=type_markers[label])
                ax2.scatter(x_umap, y_umap, color=cmap((label+1)/len(lesion_type_labels)), marker=type_markers[label])

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('t-SNE', fontsize=24)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel('UMAP', fontsize=24)

    plt.tight_layout()
    plt.savefig(results_dir + 'embedding_of_lesion_types.png', dpi=600)

    # boxplot for lesion-type importance across folds


    corresponding_lesion_type_importance, corresponding_lime_importance = [], []

    for n in range(n_lesion_types_first_fold):
        corresponding_lesion_type_importance.append([])
        corresponding_lime_importance.append([])

    for fold, (type_importances, lime_type_importances) in enumerate(zip(all_lesion_importances, all_lime_importances)):
        #types don't correspond yet
        fold_type_labels = corresponding_lesion_types[(fold)*n_lesion_types_first_fold:(fold+1)*n_lesion_types_first_fold]

        for type_number in fold_type_labels:
            corresponding_lesion_type_importance[type_number].append(type_importances[type_number])
            corresponding_lime_importance[type_number].append(lime_type_importances[type_number])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(24, 6), dpi=600)

    ax.boxplot(corresponding_lesion_type_importance)
    ax.set_ylabel('Lesion-type Gini importance', fontsize=20)
    ax.set_xlabel('Lesion-type', fontsize=20)

    ax2.boxplot(corresponding_lime_importance)
    ax2.set_ylabel('Lesion-type LIME importance', fontsize=20)
    ax2.set_xlabel('Lesion-type', fontsize=20)

    plt.tight_layout()
    plt.savefig(results_dir + 'corresponding_lesion_importance.png', bbox_inches='tight')


    fig, axes = plt.subplots(1, 3)

    data = [n_components]
    print('lesion-types:', data)
    axes[0].boxplot(data)
    axes[0].set_ylabel('Number of lesion-types', fontsize=20)

    # importance = np.zeros((n_folds, np.max(n_components)))
    # 
    # for fold, mixture_models in enumerate(bol_mixtures):
    #     importance_start_idx = 0
    # 
    #     rfs = random_forests['Placebo']
    #     lesion_importance = rfs[fold].feature_importances_
    # 
    #     sorted_indices = np.argsort(mixture_models.weights_)
    # 
    #     for c, cluster_idx in enumerate(sorted_indices):
    #         lesion_type_means[fold, :] = mixture_models.means_[cluster_idx, :]
    #         importance[fold, c] = lesion_importance[importance_start_idx+c]
    # 
    #     importance_start_idx += len(sorted_indices)
    # 
    # 
    # dim_mean = np.mean(lesion_type_means, axis=0)
    # dim_var = np.var(lesion_type_means, axis=0)
    # 
    # print('cluster centre means:', dim_mean.shape)
    # print('cluster centre variances:', dim_var.shape)
    # 
    # diffs = []
    # 
    # for fold, lesion_type_centre in enumerate(lesion_type_means):
    #     print('lesion type centre:', lesion_type_centre.size)
    # 
    #     diff = np.subtract(lesion_type_centre, dim_mean)
    #     diff_normalized = np.divide(diff, dim_var)
    # 
    #     diffs.append(diff_normalized)
    # 
    # data2 = [diffs]
    # 
    # axes[1].boxplot(data2)
    # axes[1].set_xlabel('Lesion size', fontsize=20)
    # axes[1].set_ylabel('Diff. from mean', fontsize=20)
    # 
    # 
    # data3 = [importance[:, 0], importance[:, 1], importance[:, 2], importance[:,3]]
    # axes[2].boxplot(data3)
    # axes[2].set_xlabel('Lesion size', fontsize=20)
    # axes[2].set_ylabel('P(A|BoL) Importance', fontsize=20)
    # 
    # plt.tight_layout()
    # plt.savefig(results_dir + 'cluster_numbers_lesion_centres.png', bbox_inches='tight')


def plot_activity_prediction_results(activity_truth, activity_posteriors, results_dir):
    classifier_names = ['Random Forest', '1-NN (Euclidean)', 'SVM (linear)', 'SVM ($\\chi^2$)', 'SVM (RBF)', 'MLP']
    colours = ['darkred', 'indianred', 'lightsalmon', 'darkorange', 'goldenrod', 'tan', 'k']

    for treatment in treatments:
        plt.figure(figsize=(8, 8))
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

    return


def predict_responders(args):
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

    if args.choose_k:
        features = load_data.loadAllData(mri_list, args.include_catani)
        n_lesion_types = choose_clusters(features, results_dir)
    else:
        n_lesion_types = args.k

    if args.predict_activity:
        mri_list, without_clinical = load_data.loadClinical(mri_list)

        print('We have ' + str(len(mri_list)) + ' patients who finished the study and ' + str(len(without_clinical)) + ' who did not')
        outcomes = load_data.get_outcomes(mri_list)

        mri_list = load_data.load_responders(datadir + responder_filename, mri_list)

        patient_results = {}
        for scan in mri_list:
            patient_results[scan.uid] = {}

        kf = StratifiedKFold(args.n_folds, shuffle=True, random_state=50)

        bol_mixture_models = []
        random_forests = defaultdict(list)
        deep_models = defaultdict(list)
        lime_importances = defaultdict(list)

        all_test_patients, activity_posterior, activity_truth, untreated_posterior = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        euclidean_knn_posterior, mahalanobis_knn_posterior, chi2_svm_posterior, rbf_svm_posterior, linear_svm_posterior, naive_bayes_posterior, deep_posterior =  defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

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
            raw_train_data = load_data.loadAllData(train_patients, args.include_catani)
            raw_test_data = load_data.loadAllData(mri_test, args.include_catani)

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

            for treatment in treatments:
                train_data, test_data = trainingData[treatment], testingData[treatment]
                train_outcomes, test_outcomes = load_data.get_outcomes(trainingPatientsByTreatment[treatment]), load_data.get_outcomes(
                    testingPatientsByTreatment[treatment])

                all_test_patients[treatment].append(testingPatientsByTreatment[treatment])

                if treatment == "Placebo":
                    # if args.feature_selection:
                    #     train_data, test_data, bad_types = bol_classifiers.lesion_type_selection(train_data, test_data, train_outcomes, test_outcomes, 8, results_dir)

                    (bestFeaturePredictions, placebo_rf, probPredicted) = bol_classifiers.random_forest(train_data, test_data, train_outcomes)
                    deep_probs, mlp_model, lime_importance = bol_classifiers.mlp(train_data, test_data, train_outcomes, test_outcomes, foldNum, results_dir)

                    random_forests[treatment].append(placebo_rf)
                    deep_models[treatment].append(mlp_model)
                    activity_truth[treatment].append(test_outcomes)
                    activity_posterior[treatment].append(probPredicted)
                    lime_importances[treatment].append(lime_importance)

                    svm_linear_posterior, svm_rbf_posterior, chi2svm_posterior = bol_classifiers.svms(train_data, test_data, train_outcomes)
                    knn_euclid_posterior, knn_maha_posterior = bol_classifiers.knn(train_data, train_outcomes, test_data)

                    chi2_svm_posterior[treatment].append(chi2svm_posterior)
                    rbf_svm_posterior[treatment].append(svm_rbf_posterior)
                    linear_svm_posterior[treatment].append(svm_linear_posterior)

                    euclidean_knn_posterior[treatment].append(knn_euclid_posterior)
                    mahalanobis_knn_posterior[treatment].append(knn_maha_posterior)

                    deep_posterior[treatment].append(deep_probs)

                    naive_bayes_posterior[treatment].append([])   # FIX IT

                # drugged patients
                else:
                    # if args.feature_selection:
                    #     train_data, test_data = bol_classifiers.apply_lesion_type_selection(train_data, test_data, bad_types)
                    # project onto untreated MS model (don't train)
                    (bestPreTrainedFeaturePredictions, meh, pretrainedProbPredicted) = bol_classifiers.random_forest(
                        train_data, test_data, train_outcomes, placebo_rf)

                    # new model on drugged patients
                    (bestFeaturePredictions, drug_rf, probDrugPredicted) = bol_classifiers.random_forest(train_data, test_data, train_outcomes)

                    deep_probs, mlp_model, lime_importance = bol_classifiers.mlp(train_data, test_data, train_outcomes, test_outcomes, foldNum, results_dir)

                    random_forests[treatment].append(drug_rf)
                    deep_models[treatment].append(mlp_model)
                    lime_importances[treatment].append(lime_importance)

                    svm_linear_posterior, svm_rbf_posterior, chi2svm_posterior = bol_classifiers.svms(train_data, test_data, train_outcomes)
                    knn_euclid_posterior, knn_maha_posterior = bol_classifiers.knn(train_data, train_outcomes, test_data)

                    deep_posterior[treatment].append(np.asarray(deep_probs))
                    activity_truth[treatment].append(test_outcomes)
                    activity_posterior[treatment].append(np.asarray(probDrugPredicted))
                    untreated_posterior[treatment].append(np.asarray(pretrainedProbPredicted))

                    chi2_svm_posterior[treatment].append(chi2svm_posterior)
                    rbf_svm_posterior[treatment].append(svm_rbf_posterior)
                    linear_svm_posterior[treatment].append(svm_linear_posterior)
                    euclidean_knn_posterior[treatment].append(knn_euclid_posterior)
                    mahalanobis_knn_posterior[treatment].append(knn_maha_posterior)

        activity_posteriors = [activity_posterior, euclidean_knn_posterior, linear_svm_posterior, chi2_svm_posterior, rbf_svm_posterior, deep_posterior]

        print('saving prediction results (all folds test cases)...')
        pickle.dump(activity_posteriors, open(datadir + 'posteriors.pkl', 'wb'))
        pickle.dump(all_test_patients, open(datadir + 'all_test_patients.pkl', 'wb'))
        pickle.dump(untreated_posterior, open(datadir + 'untreated_posterior.pkl', 'wb'))
        pickle.dump(activity_truth, open(datadir + 'activity_truth.pkl', 'wb'))
        pickle.dump(bol_mixture_models, open(datadir + 'mixture_models.pkl', 'wb'))
        pickle.dump(random_forests, open(datadir + 'random_forests.pkl', 'wb'))
        # pickle.dump(deep_models, open(datadir + 'deep_models.pkl', 'wb'))
        print('saved!')
    else:
        activity_posteriors = pickle.load(open(datadir + 'posteriors.pkl', 'rb'))
        all_test_patients = pickle.load(open(datadir + 'all_test_patients.pkl', 'rb'))
        untreated_posterior = pickle.load(open(datadir + 'untreated_posterior.pkl', 'rb'))
        activity_truth = pickle.load(open(datadir + 'activity_truth.pkl', 'rb'))
        bol_mixture_models = pickle.load(open(datadir + 'mixture_models.pkl', 'rb'))
        random_forests = pickle.load(open(datadir + 'random_forests.pkl', 'rb'))
        # deep_models = pickle.load(open(datadir + 'deep_models.pkl', 'rb'))

    best_p_a, best_p_d = responder_roc(all_test_patients, activity_truth, activity_posteriors[5], untreated_posterior, args.n_folds, results_dir)

    plot_activity_prediction_results(activity_truth, activity_posteriors, results_dir)

    end = time.time()
    elapsed = end - start

    cluster_stability(bol_mixture_models, random_forests, lime_importances, results_dir)
    print(str(elapsed / 60), 'minutes elapsed.')

    return experiment_number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MS Drug Responder Prediction.')
    parser.add_argument('--choose-k', type=bool, default=False, metavar='N',
                        help='choose the number of lesion-types (default: False)')
    parser.add_argument('--k', type=int, default=60, metavar='N',
                        help='if choose-k is \'False\', number of lesion-types (default: 60)')
    parser.add_argument('--predict-activity', type=bool, default=False, metavar='N',
                        help='predict activity. if false, loads pre-computed results from previous run (default: True')
    parser.add_argument('--n-folds', type=int, default=50, metavar='N',
                        help='number of folds for cross-validation (default: 50)')
    parser.add_argument('--get-features', type=bool, default=False, metavar='N',
                        help='extract features from the imaging data (default: False)')
    parser.add_argument('--feature-selection', type=bool, default=False, metavar='N',
                        help='remove lesion types that have no information (default: False)')
    parser.add_argument('--include-catani', type=bool, default=False, metavar='N',
                        help='include the Catani context priors in the features for determining lesion-types (default: False)')

    args = parser.parse_args()
    print('Arguments:', args)

    if args.get_features:
        print('Extracting features from imaging data and writing to disk')
        write_features(include_catani=False)

    experiment_number = predict_responders(args)
    print('This experiment was brought to you by the number:', experiment_number)