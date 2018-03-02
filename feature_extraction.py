import nibabel as nib
import numpy as np

from mri import mri

import pickle, csv, os, time, sys, subprocess, h5py

from random import shuffle
import skeletons

import bitstring
from multiprocessing import Pool, Process

data_dir = '/data1/users/adoyle/MS-LAQ/MS-LAQ-302-STX/'
icbmRoot = data_dir + 'quarantine/common/models/icbm_avg_152_'
lesion_atlas = data_dir + 'quarantine/common/models/icbm_avg_3714_t2les.mnc.gz'

reload_list = False

modalities = ['t1p', 't2w', 'pdw', 'flr']

thinners = skeletons.thinningElements()

lbpRadii = [1]

def write_clinical_outputs(mri_list):
    # csvwriter = csv.writer(open(data_dir + 'extraOnes.csv', 'w'))
    csvreader = csv.reader(open(data_dir + 'MSLAQ-clinical.csv'))

    index = 0
    for row in csvreader:
        if index >= 8:

            saveDocument = {}
            uid = row[0][0:3] + row[0][4:]
            treatment = row[4]

            newT2 = row[29]
            newT1 = row[32]
            atrophy = row[36]

            inList = False
            for scan in mri_list:
                if scan.uid == uid:
                    inList = True
                    right_scan = scan

            if not inList:  # we don't have imaging data for the results, log it
                print(uid, 'NOT FOUND')
                # csvwriter.writerow([uid[0:3] + '_' + uid[4:]])
            else:
                print(uid, treatment, newT2, newT1, atrophy)
                saveDocument['treatment'] = treatment
                try:
                    saveDocument['newT1'] = int(newT1)
                except ValueError:
                    saveDocument['newT1'] = 0
                try:
                    saveDocument['newT2'] = int(newT2)
                except ValueError:
                    saveDocument['newT2'] = 0

                try:
                    saveDocument['atrophy'] = float(atrophy)
                except:
                    saveDocument['atrophy'] = 0.0

                print(right_scan.features_dir + 'clinical.pkl')
                pickle.dump(saveDocument, open(right_scan.features_dir + 'clinical.pkl', 'wb'))

        index += 1

def separate_lesions(scan):
    lesion_image = nib.load(scan.lesions).get_data()
    lesion_locations = list(np.asarray(np.nonzero(lesion_image)).T)
    connected_lesion = np.zeros((len(lesion_locations)))

    lesion_list = []
    for i, (x, y, z) in enumerate(lesion_locations):
        for lesion in lesion_list:
            for point in lesion:
                if np.abs(x - point[0]) <= 1 and np.abs(y - point[1]) <= 1 and np.abs(z - point[2]) <= 1:
                    lesion.append([x, y, z])
                    connected_lesion[i] = True
                if connected_lesion[i]:
                    break

        if not connected_lesion[i]:
            newLesion = [[x, y, z]]
            lesion_list.append(newLesion)

    return lesion_list


def uniformLBP(image, lesion, radius):
    lbp = bitstring.BitArray('0b00000000')

    r = radius
    uniformPatterns = np.zeros(9, dtype='float32')

    for i, [x, y, z] in enumerate(lesion):
        threshold = image[x, y, z]

        lbp.set(image[x, y - r, z] > threshold, 0)
        lbp.set(image[x, y - r, z + r] > threshold, 1)
        lbp.set(image[x, y, z + r] > threshold, 2)
        lbp.set(image[x, y + r, z + r] > threshold, 3)
        lbp.set(image[x, y + r, z] > threshold, 4)
        lbp.set(image[x, y + r, z - r] > threshold, 5)
        lbp.set(image[x, y, z - r] > threshold, 6)
        lbp.set(image[x, y - r, z - r] > threshold, 7)

        transitions = 0
        for bit in range(len(lbp) - 1):
            if not lbp[bit] == lbp[bit + 1]:
                transitions += 1

        if not lbp[0] == lbp[-1]:
            transitions += 1

        ones = lbp.count(1)

        if transitions <= 2:
            uniformPatterns[ones] += 1.0 / float(len(lesion))
        else:
            uniformPatterns[8] += 1.0 / float(len(lesion))

    return uniformPatterns


def get_rift(scan, img):
    numBinsTheta = 4
    sigma = np.sqrt(2)

    binsTheta = np.linspace(0, 2 * np.pi, num=numBinsTheta + 1, endpoint=True)

    grad_x, grad_y, grad_z = {}, {}, {}
    mag, theta = {}, {}

    for mod in modalities:
        grad_x[mod], grad_y[mod], grad_z[mod] = np.gradient(img[mod])

        mag[mod] = np.sqrt(np.square(grad_x[mod]) + np.square(grad_y[mod]))
        theta[mod] = np.arctan2(grad_y[mod], grad_x[mod])

    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)

        for mod in modalities:
            feature = np.zeros((numBinsTheta))

            lesion_points = np.asarray(lesion)
            # print('lesion points:', lesion_points.shape)
            # for point in lesion_points:
            #     print(point)

            x_min, x_max = np.min(lesion_points[:, 0]), np.max(lesion_points[:, 0])
            # print(x_min, x_max)

            for xc in range(x_min, x_max+1):
                in_plane = lesion_points[lesion_points[:, 0] == xc]
                yc = int(np.mean(in_plane[:, 1]))
                zc = int(np.mean(in_plane[:, 2]))

                gradient_direction, gradient_strength = [], []
                for p, evalPoint in enumerate(in_plane):
                    x = xc
                    y = yc + evalPoint[0]
                    z = zc + evalPoint[1]

                    if [x, y, z] in lesion:
                        relTheta = np.arctan2((y - yc), (z - zc))
                        outwardTheta = (theta[mod][x, y, z] - relTheta + 2 * np.pi) % (2 * np.pi)

                        gradient_direction.append(outwardTheta)
                        gradient_strength.append(mag[mod][x, y, z] / 1000)

                    # gaussian = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
                    #     - (np.square(y - yc) + np.square(z - zc)) / (2 * sigma ** 2))

                hist, bins = np.histogram(gradient_direction, bins=binsTheta, range=(0, np.pi),
                                          weights=gradient_strength)

                if not np.isnan(np.min(hist)):
                    feature += hist / float(len(in_plane))
                else:
                    print('NaNs in RIFT!')

            saveDocument[mod] = feature

        pickle.dump(saveDocument, open(scan.features_dir + 'rift_' + str(l) + '.pkl', "wb"))


def loadMRIList():
    complete_data_subjects, missing_data_subjects = 0, 0

    mri_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('_m0_t1p.mnc.gz'):
                scan = mri(f)

                if os.path.isfile(scan.lesions):
                    if os.path.isfile(scan.images['t1p']) and os.path.isfile(scan.images['t2w']) and os.path.isfile(
                            scan.images['pdw']) and os.path.isfile(scan.images['flr']):
                        print('Parsing files for', f)
                        mri_list.append(scan)
                        complete_data_subjects += 1
                    else:
                        print('Missing MRI modality: ', f)
                        missing_data_subjects += 1
                else:
                    print('Missing lesion labels: ', f)
                    missing_data_subjects += 1

    print(complete_data_subjects, '/', missing_data_subjects + complete_data_subjects,
          'have all modalities and lesion labels')

    mri_list_lesions = []
    for i, scan in enumerate(mri_list):
        scan.lesionList = separate_lesions(scan)
        mri_list_lesions.append(scan)
        print(scan.uid, i + 1, '/', len(mri_list) + 1)

    return mri_list_lesions


def get_context(scan, images):
    # contextMin = {"csf": -0.001, "wm": -0.001, "gm": -0.001, "pv": -0.001, "lesion": -0.001}
    # contextMax = {'csf': 1.001, 'wm': 1.001, 'gm': 1.001, 'pv': 1.001, 'lesion': 0.348}
    #
    # numBins = 4

    wm_tracts = ['Anterior_Segment', 'Arcuate', 'Cingulum', 'Cortico_Ponto_Cerebellum', 'Cortico_Spinal',
                 'Inferior_Cerebellar_Pedunculus', 'Inferior_Longitudinal_Fasciculus',
                 'Inferior_Occipito_Frontal_Fasciculus', 'Long_Segment', 'Optic_Radiations', 'Posterior_Segment',
                 'Superior_Cerebelar_Pedunculus', 'Uncinate', 'Anterior_Commissure', 'Corpus_Callosum', 'Fornix',
                 'Internal_Capsule']

    for tissue in scan.tissues:
        filename = scan.priors[tissue]
        images[tissue] = nib.load(filename).get_data()

    for wm_tract in wm_tracts:
        images[wm_tract] = nib.load('/data1/users/adoyle/atlases/Catani/MSLAQ/' + wm_tract + '.nii').get_data()

    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)

        for tissue in scan.tissues + wm_tracts:
            context = []

            for p in lesion:
                context.append(images[tissue][p[0], p[1], p[2]])

            saveDocument[tissue] = [np.mean(context), np.var(context)]

        pickle.dump(saveDocument, open(scan.features_dir + 'context_' + str(l) + '.pkl', "wb"))


def get_lbp(scan, images):
    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)

        feature = np.zeros((len(lbpRadii), 9))

        for j, mod in enumerate(modalities):
            saveDocument[mod] = {}

            for r, radius in enumerate(lbpRadii):
                feature[r, ...] = uniformLBP(images[mod], lesion, radius)
            saveDocument[mod] = feature

        pickle.dump(saveDocument, open(scan.features_dir + 'lbp_' + str(l) + '.pkl', "wb"))


def get_intensity(scan, images):
    # intensityMin = {"t1p": 32.0, "t2w": 10.0, "flr": 33.0, "pdw": 49.0}
    # intensityMax = {'t1p': 1025.0, 't2w': 1000.0, 'flr': 1016.0, 'pdw': 1018.0}

    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)

        for m in modalities:
            intensities = []
            for point in lesion:
                intensities.append(images[m][point[0], point[1], point[2]] / 1000)

            # intensityHist = np.histogram(intensities, histBins, (intensityMin[m], intensityMax[m]))
            # intensityHist = intensityHist[0] / np.sum(intensityHist[0], dtype='float')
            #
            # if np.isnan(intensityHist).any():
            #     intensityHist = np.zeros((histBins))
            #     intensityHist[0] = 1

            saveDocument[m] = [np.mean(intensities), np.var(intensities)]

        pickle.dump(saveDocument, open(scan.features_dir + 'intensity_' + str(l) + '.pkl', "wb"))


def getFeaturesOfList(mri_list):
    for i, scan in enumerate(mri_list):
        images = {}
        for j, m in enumerate(modalities):
            images[m] = nib.load(scan.images[m]).get_data()

        print('Patient:', scan.uid, i + 1, '/', len(mri_list) + 1)
        startTime = time.time()

        get_context(scan, images)
        get_lbp(scan, images)
        get_rift(scan, images)
        get_intensity(scan, images)

        elapsed = time.time() - startTime
        print(elapsed, "seconds")


def chunks(l, n):
    shuffle(l)
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    startTime = time.time()

    print('Loading MRI file list...')

    if reload_list:
        mri_list = loadMRIList()
        outfile = open(data_dir + 'mri_list.pkl', 'wb')
        pickle.dump(mri_list, outfile)
        outfile.close()
        print('Cached MRI file listing')
    else:
        infile = open(data_dir + 'mri_list.pkl', 'rb')
        mri_list = pickle.load(infile)
        infile.close()

    print('MRI list loaded')

    print('extracting imaging ')
    getFeaturesOfList(mri_list)

    print('writing clinical outputs...')
    write_clinical_outputs(mri_list)
    print('Done')

    endTime = time.time()

    elapsed = endTime - startTime
    print("Total time elapsed:", elapsed / 3600, 'hours', elapsed / 60, 'minutes')


if __name__ == "__main__":
    main()
