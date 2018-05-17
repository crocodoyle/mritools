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

from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras import backend as K

import lime
import lime.lime_tabular

modalities = ['t1p', 't2w', 'pdw', 'flr']
tissues = ['csf', 'wm', 'gm', 'pv', 'lesion']
metrics = ['newT2']
feats = ["Context", "RIFT", "LBP", "Intensity"]

scoringMetrics = ['TP', 'FP', 'TN', 'FN']

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


def mlp(train_data, test_data, train_outcomes, test_outcomes, fold_num, results_dir):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model_checkpoint = ModelCheckpoint(results_dir + "fold_" + str(fold_num) + "_best_weights.hdf5", monitor="categorical_accuracy", save_best_only=True)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # model.summary()

    train_labels = to_categorical(train_outcomes, num_classes=2)
    test_labels = to_categorical(test_outcomes, num_classes=2)

    hist = model.fit(train_data, train_labels, batch_size=128, epochs=1200, validation_data=(test_data, test_labels), callbacks=[model_checkpoint], verbose=False)

    # print(model.metrics_names)

    model.load_weights(results_dir + "fold_" + str(fold_num) + "_best_weights.hdf5")
    model.save(results_dir + 'best_bol_model' + str(fold_num) + '.hdf5')

    deep_probabilities = model.predict_proba(test_data)

    train_labels = to_categorical(train_outcomes, num_classes=2)

    explainer = lime.lime_tabular.LimeTabularExplainer(train_data, training_labels=train_labels, discretize_continuous=True, discretizer='quartile', class_names=['Inactive', 'Active'])

    lime_type_importance = np.zeros((train_data.shape[1]))

    for i in range(test_data.shape[0]):
        prediction = model.predict(test_data[i, ...][np.newaxis, ...])
        if prediction[0][1] > 0.5:
            prediction = 1
            label = ['Active']
        else:
            prediction = 0
            label = ['Inactive']

        if test_outcomes[i] == prediction:
            exp = explainer.explain_instance(test_data[i, ...], model.predict_proba, num_features=10, labels=[prediction])
            exp.save_to_file(results_dir + 'explanation' + str(fold_num) + '-' + str(i) + '.html')
            important_types = exp.as_map()
            # print('types', important_types)

            fig = exp.as_pyplot_figure(label=prediction)
            fig.savefig(results_dir + str(fold_num) + '_' + str(i) + '_explained.png')

            for lesion_type in important_types[prediction]:
                lime_type_importance[lesion_type[0]] += lesion_type[1]

    K.clear_session()

    return deep_probabilities, model, lime_type_importance


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


def lesion_type_selection(trainData, testData, trainOutcomes, testOutcomes, minTypes, results_dir):
    train = trainData
    test = testData    
    
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=2000, n_jobs=-1, oob_score=True)
    rf.fit(trainData, trainOutcomes)
    
    allFeatureImportance = rf.feature_importances_
    featureImportance = allFeatureImportance
 
    typesLeft = len(featureImportance)
    
    trainScores, oobScores, testScores, numFeatures = [], [], [], []

    while typesLeft > minTypes:
        removeThisRound = []
        
        for r in range(int(np.ceil(0.1*typesLeft))):
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

        print("out of bag scores", oobScores)

    best_number = numFeatures[np.argmax(np.asarray(oobScores))]

    print(best_number, 'is the optimal number of features')
    
    plt.figure()
    plt.plot(numFeatures, oobScores, label="Out-of-Bag Score")
    plt.plot(numFeatures, trainScores, label="Training Score")
    plt.plot(numFeatures, testScores, label="Test Score")
    plt.xlabel('Number of codewords in BoL', fontsize=20)
    plt.ylabel('Mean Accuracy', fontsize=20)
    plt.legend(shadow=True, fontsize=20)
    plt.tight_layout()
    plt.savefig(results_dir + 'feature_selection.png', dpi=500)
    plt.close()

    train = trainData
    test = testData
    finalRemove = np.shape(testData)[1] - best_number
    
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


def apply_lesion_type_selection(train_data, test_data, types_to_remove):
    for remove in types_to_remove:
        train_data = np.delete(train_data, remove, 1)
        test_data = np.delete(test_data, remove, 1)

    return train_data, test_data
