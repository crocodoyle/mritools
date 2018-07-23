import nibabel as nib
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    csvreader = csv.reader(open(data_dir + '2018-01_BRAVO_IPMSA.csv'))
    lines = list(csvreader)

    for scan in mri_list:
        saveDocument = {}
        patient_id = scan.uid[4:]

        for row in lines:
            if patient_id in row[2]:
                treatment = row[5].split(' ')[0]
                newT2 = row[42]
                relapse = row[32]

                gad12 = row[43]
                gad24 = row[44]

                if not 'NULL' in gad12 and not 'NULL' in gad24:
                    gad = int(gad12) + int(gad24)
                    saveDocument['gad'] = str(gad)

                country = row[4]
                race = row[8]
                sex = row[9]
                age = row[11]

                saveDocument['newT2'] = newT2
                saveDocument['relapse'] = relapse
                saveDocument['treatment'] = treatment

                saveDocument['country'] = country
                saveDocument['race'] = race
                saveDocument['sex'] = sex
                saveDocument['age'] = age

                print(scan.uid, saveDocument)


        pickle.dump(saveDocument, open(scan.features_dir + 'clinical.pkl', 'wb'))

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
    r = radius
    uniformPatterns = np.zeros(9, dtype='float32')

    for i, [x, y, z] in enumerate(lesion):
        threshold = image[x, y, z]

        lbp = bitstring.BitArray('0b00000000')

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
    visualize_slice = False
    visualize_lesion = False

    binsTheta = np.linspace(0, 2 * np.pi, num=numBinsTheta + 1, endpoint=True)

    grad_x, grad_y, grad_z = {}, {}, {}
    mag, theta = {}, {}

    for mod in modalities:
        grad_x[mod], grad_y[mod], grad_z[mod] = np.gradient(img[mod])

        mag[mod] = np.sqrt(np.square(grad_y[mod]) + np.square(grad_z[mod]))
        theta[mod] = np.arctan2(grad_z[mod], grad_y[mod])

    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)

        for mod in modalities:
            feature = np.zeros(numBinsTheta, dtype='float32')

            lesion_points = np.asarray(lesion)
            # print('lesion points:', lesion_points.shape)
            # for point in lesion_points:
            #     print(point)

            x_min, x_max = np.min(lesion_points[:, 0]), np.max(lesion_points[:, 0])
            # print('Lesion connected across', x_max - x_min, 'slices')

            for xc in range(x_min, x_max+1):
                in_plane = lesion_points[lesion_points[:, 0] == xc]

                yc = np.mean(in_plane[:, 1])
                zc = np.mean(in_plane[:, 2])

                # print('Lesion has', len(in_plane), 'voxels in slice', xc, 'centered at', yc, zc)

                if len(in_plane) > 10 and np.random.rand() > 0.99 and not visualize_lesion:
                    visualize_slice = True
                    visualize_lesion = True

                if visualize_slice:
                    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 8))

                    img = nib.load(scan.images['t2w']).get_data()
                    lesionMaskImg = np.zeros((np.shape(img)))

                    angle = np.zeros_like(theta['t2w'])
                    magnitude = np.zeros_like(mag['t2w'])

                    for point in lesion:
                        lesionMaskImg[point[0], point[1], point[2]] = 1

                        angle[point[0], point[1], point[2]] = theta['t2w'][point[0], point[1], point[2]]
                        magnitude[point[0], point[1], point[2]] = mag['t2w'][point[0], point[1], point[2]]

                    maskImg = np.ma.masked_where(lesionMaskImg == 0, np.ones((np.shape(lesionMaskImg))) * 5000)

                    maskSquare = np.zeros((np.shape(img)))
                    maskSquare[int(xc), int(yc) - 20:int(yc) + 20, int(zc) - 20] = 1
                    maskSquare[int(xc), int(yc) - 20:int(yc) + 20, int(zc) + 20] = 1
                    maskSquare[int(xc), int(yc) - 20, int(zc) - 20:int(zc) + 20] = 1
                    maskSquare[int(xc), int(yc) + 20, int(zc) - 20:int(zc) + 20] = 1

                    square = np.ma.masked_where(maskSquare == 0, np.ones(np.shape(maskSquare)) * 5000)

                    lesionMaskPatch = maskImg[int(xc), int(yc) - 20:int(yc) + 20, int(zc) - 20:int(zc) + 20]

                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.imshow(img[int(xc), 20:200, 20:175], cmap=plt.cm.gray, interpolation='nearest', origin='lower')
                    ax1.imshow(maskImg[int(xc), 20:200, 20:175], cmap=plt.cm.autumn, interpolation='nearest', alpha=0.25, origin='lower')
                    ax1.imshow(square[int(xc), 20:200, 20:175], cmap=plt.cm.autumn, interpolation='nearest', origin='lower')

                    centre_point = (20, 20)

                    ax2.imshow(img[int(xc), int(yc) - 20:int(yc) + 20, int(zc) - 20:int(zc) + 20], cmap=plt.cm.gray, interpolation='nearest', origin='lower')
                    ax2.imshow(lesionMaskPatch, cmap=plt.cm.autumn, alpha=0.25, interpolation='nearest', origin='lower')
                    ax2.set_xticks([])
                    ax2.set_yticks([])

                    mag_img = ax3.imshow(magnitude[int(xc), int(yc) - 20: int(yc) + 20, int(zc) - 20: int(zc) + 20], cmap=plt.cm.gray, interpolation='nearest', origin='lower')

                    divider = make_axes_locatable(ax3)
                    cax = divider.append_axes("right", size="5%", pad=0.05)

                    plt.colorbar(mag_img, cax=cax)
                    ax3.set_xticks([])
                    ax3.set_yticks([])

                    max_grad = np.argmax(magnitude[int(xc), int(yc) - 20: int(yc) + 20, int(zc) - 20: int(zc) + 20])

                    max_grad_pos = np.unravel_index(max_grad, magnitude[int(xc), int(yc) - 20: int(yc) + 20, int(zc) - 20: int(zc) + 20].shape)

                    max_grad_val = magnitude[int(xc), int(yc) - 20: int(yc) + 20, int(zc) - 20: int(zc) + 20][max_grad_pos]
                    max_grad_angle = angle[int(xc), int(yc) - 20: int(yc) + 20, int(zc) - 20: int(zc) + 20][max_grad_pos]

                    arrow_angle = max_grad_angle + np.arctan2((max_grad_pos[0] - yc), (max_grad_pos[1] - zc))

                    o = np.sin(arrow_angle)*(max_grad_val / 100)*5
                    a = np.cos(arrow_angle)*(max_grad_val / 100)*5

                    arrow_begin = (max_grad_pos[1], max_grad_pos[0])
                    arrow_end = (a, o)

                    # print('arrow begin:', arrow_begin, 'arrow end:', arrow_end)

                    ax4.imshow(img[int(xc), int(yc) - 20:int(yc) + 20, int(zc) - 20:int(zc) + 20], cmap=plt.cm.gray, interpolation='nearest', origin='lower')
                    ax4.imshow(lesionMaskPatch, cmap=plt.cm.autumn, alpha=0.25, interpolation='nearest', origin='lower')

                    ax4.arrow(arrow_begin[0], arrow_begin[1], arrow_end[0], arrow_end[1], head_width=2, head_length=2, color='b')

                    ax4.plot(centre_point[0], centre_point[1], 'ro', markersize=2)
                    ax4.plot(arrow_begin[0], arrow_begin[1], 'bo', markersize=2)

                    radial_line_x = [centre_point[0], arrow_begin[0]]
                    radial_line_y = [centre_point[1], arrow_begin[1]]

                    ax4.plot(radial_line_x, radial_line_y, color='r')

                    ax4.set_xticks([])
                    ax4.set_yticks([])

                    visualize_slice = False

                gradient_direction, gradient_strength = [], []
                for (x, y, z) in in_plane:
                    # print('Point:', x, y, z)

                    if not y == yc and not z == zc:
                        relTheta = np.arctan2((z - zc), (y - yc))
                        outwardTheta = (theta[mod][x, y, z] - relTheta + 2 * np.pi) % (2 * np.pi)

                        # print('Relative angle:', relTheta)
                        # print('Angle from radius:', outwardTheta)

                        gradient_direction.append(outwardTheta)
                        gradient_strength.append(mag[mod][x, y, z])

                    # gaussian = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
                    #     - (np.square(y - yc) + np.square(z - zc)) / (2 * sigma ** 2))

                hist, bins = np.histogram(gradient_direction, bins=binsTheta, range=(0, np.pi),
                                          weights=gradient_strength)

                # print('Histogram values, bins:', hist, bins)
                feature += hist / (x_max - x_min + 1)

                if visualize_lesion:
                    ax5.bar(bins[:-1], hist)
                    ax5.set_xticks(list(np.linspace(0, 2*np.pi, num=4, endpoint=False)))
                    ax5.set_xticklabels(['inward', 'left', 'outward', 'right'])
                    ax5.set_yticks([])

                    plt.savefig(data_dir + '/examples/' + 'RIFT_example_' + str(scan.uid) + '_lesion_' + str(l) + '.png')
                    plt.close()
                    visualize_slice = False
                    visualize_lesion = False

            saveDocument[mod] = feature / 1000

        # print('Final RIFT descriptor:', saveDocument)
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


def get_context(scan, images, include_catani):
    # contextMin = {"csf": -0.001, "wm": -0.001, "gm": -0.001, "pv": -0.001, "lesion": -0.001}
    # contextMax = {'csf': 1.001, 'wm': 1.001, 'gm': 1.001, 'pv': 1.001, 'lesion': 0.348}
    #
    # numBins = 4

    wm_tracts = ['Anterior_Segment', 'Arcuate', 'Cingulum', 'Cortico_Ponto_Cerebellum', 'Cortico_Spinal',
                 'Inferior_Cerebellar_Pedunculus', 'Inferior_Longitudinal_Fasciculus',
                 'Inferior_Occipito_Frontal_Fasciculus', 'Long_Segment', 'Optic_Radiations', 'Posterior_Segment',
                 'Superior_Cerebelar_Pedunculus', 'Uncinate', 'Anterior_Commissure', 'Corpus_Callosum', 'Fornix',
                 'Internal_Capsule']

    wm_networks = ['Projection', 'Cerebellar', 'Optic', 'Cingulum', 'Inferior', 'Arcuate', 'Perisylvian', 'Anterior_Commissure', 'Fornix', 'Corpus_Callosum']

    for tissue in scan.tissues:
        filename = scan.priors[tissue]
        images[tissue] = nib.load(filename).get_data()

    for wm_network in wm_networks:
        images[wm_network] = nib.load('/data1/users/adoyle/atlases/Catani/MSLAQ/' + wm_network + '.nii').get_data()

    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)

        context_priors = scan.tissues
        if include_catani:
            context_priors += wm_tracts

        for tissue in context_priors:
            context = []

            for p in lesion:
                context.append(images[tissue][p[0], p[1], p[2]])

            saveDocument[tissue] = [np.mean(context), np.var(context)]

        pickle.dump(saveDocument, open(scan.features_dir + 'context_' + str(l) + '.pkl', "wb"))


def get_lbp(scan, images):
    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)

        for j, mod in enumerate(modalities):
            feature = np.zeros((len(lbpRadii), 9))

            for r, radius in enumerate(lbpRadii):
                feature[r, ...] = uniformLBP(images[mod], lesion, radius)
            # print(mod, feature)
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

            saveDocument[m] = [np.mean(intensities), np.var(intensities), np.min(intensities), np.max(intensities)]

        pickle.dump(saveDocument, open(scan.features_dir + 'intensity_' + str(l) + '.pkl', "wb"))


def getFeaturesOfList(mri_list, include_catani):
    for i, scan in enumerate(mri_list):
        images = {}
        for j, m in enumerate(modalities):
            images[m] = nib.load(scan.images[m]).get_data()

        print('Patient:', scan.uid, i + 1, '/', len(mri_list) + 1)
        startTime = time.time()

        get_context(scan, images, include_catani)
        get_lbp(scan, images)
        get_rift(scan, images)
        get_intensity(scan, images)

        elapsed = time.time() - startTime
        print(elapsed, "seconds")


def chunks(l, n):
    shuffle(l)
    for i in range(0, len(l), n):
        yield l[i:i + n]


def write_features(include_catani=True):
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
    getFeaturesOfList(mri_list, include_catani)

    print('writing clinical outputs...')
    write_clinical_outputs(mri_list)
    print('Done')

    endTime = time.time()

    elapsed = endTime - startTime
    print("Total time elapsed:", elapsed / 60, 'minutes')
    return


if __name__ == "__main__":
    write_features()
