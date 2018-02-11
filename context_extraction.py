import nibabel as nib
import numpy as np
import os

from mri import mri

import pickle, csv

import subprocess
import time, sys

from random import shuffle
import skeletons

import bitstring
from multiprocessing import Pool, Process

data_dir = '/data1/users/adoyle/MS-LAQ/MS-LAQ-302-STX/'
icbmRoot = data_dir + 'quarantine/common/models/icbm_avg_152_'
lesion_atlas = data_dir + 'quarantine/common/models/icbm_avg_3714_t2les.mnc.gz'


threads = 8
recompute = True
reconstruct = False

doLBP = True
doContext = True
doRIFT = True
doIntensity = True

reload_list = True

modalities = ['t1p', 't2w', 'pdw', 'flr']

riftRadii = [3, 6]
lbpRadii = [1]

lbpBinsTheta = 6
lbpBinsPhi = 4

thinners = skeletons.thinningElements()

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


def convertToNifti(mri_list):
    new_list = []
    for scan in mri_list:
        for mod in modalities:
            if not '.nii' in scan.images[mod]:
                subprocess.call(['mnc2nii', scan.images[mod], scan.images[mod][0:-7] + '.nii'])
                scan.images[mod] = scan.images[mod][0:-7] + '.nii'
                subprocess.call(['gzip', scan.images[mod]])
                scan.images[mod] += '.gz'                
                            
        for prior in scan.tissues:
            if not '.nii' in scan.priors[prior]:
                subprocess.call(['mnc2nii', scan.priors[prior], scan.priors[prior][0:-7]+'.nii'])
                scan.priors[prior] = scan.priors[prior][0:-7]+'.nii'
                subprocess.call(['gzip', scan.priors[prior]])
                scan.priors[prior] += '.gz'
                
        new_list.append(scan)
    
    outfile = open('/usr/local/data/adoyle/new_mri_list.pkl', 'wb')
    pickle.dump(new_list, outfile)
    outfile.close()
    
    return new_list


def invertLesionCoordinates(mri_list):
    new_list = []
    for scan in mri_list:
        new_lesion_list = []
        for lesion in scan.lesionList:
            new_lesion = []
            for point in lesion:
                x = point[2]
                y = point[1]
                z = point[0]
                new_lesion.append([x, y, z])
            new_lesion_list.append(new_lesion)
        scan.lesionList = new_lesion_list
        new_list.append(scan)
    
    outfile = open('/usr/local/data/adoyle/new_mri_list.pkl', 'wb')
    pickle.dump(new_list, outfile)
    outfile.close()    

    return new_list


def getBoundingBox(mri_list):
    lesTypes = ['tiny', 'small', 'medium', 'large']

    boundingBoxes, xMax, yMax, zMax = {}, {}, {}, {}

    for lesType in lesTypes:
        xMax[lesType] = 0
        yMax[lesType] = 0
        zMax[lesType] = 0
    
    for scan in mri_list:
        for les in scan.lesionList:
            if len(les) > 2 and len(les) < 11:
                lesType = 'tiny'
            if len(les) > 10 and len(les) < 26:
                lesType = 'small'
            if len(les) > 25 and len(les) < 101:
                lesType = 'medium'
            if len(les) > 100:
                lesType = 'large'
            if len(les) < 3:
                continue

            lesion = np.asarray(les)
            xRange = np.amax(lesion[:,0]) - np.amin(lesion[:,0])
            yRange = np.amax(lesion[:,1]) - np.amin(lesion[:,1])
            zRange = np.amax(lesion[:,2]) - np.amin(lesion[:,2])

            if xRange > xMax[lesType]:
                xMax[lesType] = xRange
            if yRange > yMax[lesType]:
                yMax[lesType] = yRange
            if zRange > zMax[lesType]:
                zMax[lesType] = zRange
    
    for lesType in lesTypes:
        boundingBoxes[lesType] = [xMax[lesType], yMax[lesType], zMax[lesType]]
        
    print('boundingBoxes: ', boundingBoxes)
    return boundingBoxes


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
    
    size = ""
    if (len(lesion) > 2) and (len(lesion) < 11):
        size = 'tiny'
    elif (len(lesion) > 10) and (len(lesion) < 26):
        size = 'small'
    elif (len(lesion) > 25) and (len(lesion) < 101):
        size = 'medium'
    elif (len(lesion) > 100):
        size = 'large'
    
    r = radius
    
    if size == 'tiny' or size == 'small':
        uniformPatterns = np.zeros(9, dtype='float32')
        
        for i, [x,y,z] in enumerate(lesion):
            threshold = image[x,y,z]
            
            lbp.set(image[x-r, y, z] > threshold, 0)
            lbp.set(image[x-r, y+r, z] > threshold, 1)
            lbp.set(image[x, y+r, z] > threshold, 2)
            lbp.set(image[x+r, y+r, z] > threshold, 3)
            lbp.set(image[x+r, y, z] > threshold, 4)
            lbp.set(image[x+r, y-r, z] > threshold, 5)
            lbp.set(image[x, y-r, z] > threshold, 6)
            lbp.set(image[x-r, y-r, z] > threshold, 7)
        
            transitions = 0
            for bit in range(len(lbp)-1):
                if not lbp[bit] == lbp[bit+1]:
                    transitions += 1
    
            if not lbp[0] == lbp[-1]:
                transitions += 1
                
            ones = lbp.count(1)
            
            if transitions <= 2:
                uniformPatterns[ones] += 1.0 / float(len(lesion))
            else:
                uniformPatterns[8] += 1.0 / float(len(lesion))
                
    elif size == 'medium' or size == 'large':
        uniformPatterns = np.zeros(9, dtype='float32')
        # garbage, skeleton = skeletons.hitOrMissThinning(lesion, thinners)

        for i, [x,y,z] in enumerate(lesion):
            threshold = image[x,y,z]

            lbp.set(image[x-r, y, z] > threshold, 0)
            lbp.set(image[x-r, y+r, z] > threshold, 1)
            lbp.set(image[x, y+r, z] > threshold, 2)
            lbp.set(image[x+r, y+r, z] > threshold, 3)
            lbp.set(image[x+r, y, z] > threshold, 4)
            lbp.set(image[x+r, y-r, z] > threshold, 5)
            lbp.set(image[x, y-r, z] > threshold, 6)
            lbp.set(image[x-r, y-r, z] > threshold, 7)

            transitions = 0
            for bit in range(len(lbp)-1):
                if not lbp[bit] == lbp[bit+1]:
                    transitions += 1

            if not lbp[0] == lbp[-1]:
                transitions += 1

            ones = lbp.count(1)

            if transitions <= 2:
                uniformPatterns[ones] += 1.0 / float(len(lesion))
            else:
                uniformPatterns[8] += 1.0 / float(len(lesion))
                
    return uniformPatterns

    
def generateRIFTRegions2D(radii):
    pointLists = []
    
    for r in range(len(radii)):
        pointLists.append([])
        
    for x in range(-np.max(radii), np.max(radii)):
        for y in range(-np.max(radii), np.max(radii)):
            distance = np.sqrt(x**2 + y**2)
            
            if distance <= radii[0]:
                pointLists[0].append([x, y])
            elif distance > radii[0] and distance <= radii[1]:
                pointLists[1].append([x, y])
            # if distance > radii[1] and distance <= radii[2]:
            #     pointLists[2].append([x, y])

    return pointLists
    
def getRIFTFeatures2D(scan, riftRegions, img):
    numBinsTheta = 4
    sigma = np.sqrt(2)
    
    binsTheta = np.linspace(0, 2*np.pi, num=numBinsTheta+1, endpoint=True)
    
    grad_x, grad_y, grad_z = {}, {}, {}
    mag, theta = {}, {}
    
    for mod in modalities:
        grad_x[mod], grad_y[mod], grad_z[mod] = np.gradient(img[mod])
    
        mag[mod] = np.sqrt(np.square(grad_x[mod]) + np.square(grad_y[mod]))
        theta[mod] = np.arctan2(grad_y[mod], grad_x[mod])
        
    for l, lesion in enumerate(scan.lesionList):
        size = ""
        if (len(lesion) > 2) and (len(lesion) < 11):
            size = 'tiny'
        elif (len(lesion) > 10) and (len(lesion) < 26):
            size = 'small'
        elif (len(lesion) > 25) and (len(lesion) < 101):
            size = 'medium'
        elif (len(lesion) > 100):
            size = 'large'
        else:
            continue
        
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)        

        for mod in modalities:
            if size == 'tiny' or size == 'small':
                feature = np.zeros((len(riftRadii), numBinsTheta))
                for pIndex, point in enumerate(lesion):
                    xc, yc, zc = point
                    for r, region in enumerate(riftRegions):
                        gradient_direction, gradient_strength = [], []
                        for p, evalPoint in enumerate(region):
                            x = xc + evalPoint[0]
                            y = yc + evalPoint[1]
                            z = zc

                            relTheta = np.arctan2((y - yc), (x - xc))
                            outwardTheta = (theta[mod][x,y,z] - relTheta + 2*np.pi)%(2*np.pi)
                            gaussianWindow = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.square(y-yc) + np.square(x-xc)) / (2 * sigma**2))

                            gradient_direction.append(outwardTheta)
                            gradient_strength.append(mag[mod][x,y,z]*gaussianWindow)

                        hist, bins = np.histogram(gradient_direction, bins=binsTheta, range=(0, np.pi), weights=gradient_strength)
                        # hist = np.divide(hist, sum(hist))
                        if not np.isnan(np.min(hist)):
                            feature[r, :] += hist / float(len(lesion))
                        else:
                            print('NaNs in RIFT for', scan.uid, 'at radius', str(riftRadii[r]))
  
            elif size == 'medium' or size == 'large':
                feature = np.zeros((len(riftRadii), numBinsTheta))
                # garbage, skeleton = skeletons.hitOrMissThinning(lesion, thinners)

                for pIndex, point in enumerate(lesion):
                    xc, yc, zc = point
                    for r, region in enumerate(riftRegions):
                        for p, evalPoint in enumerate(region):
                            gradient_direction, gradient_strength = [], []

                            x = xc + evalPoint[0]
                            y = yc + evalPoint[1]
                            z = zc

                            if [x, y, z] in lesion:
                                relTheta = np.arctan2((y - yc), (x - xc))
                                outwardTheta = (theta[mod][x, y, z] - relTheta + 2 * np.pi) % (2 * np.pi)

                                gradient_direction.append(outwardTheta)
                                gradient_strength.append(mag[mod][x, y, z])

                            gaussianWindow = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (np.square(y - yc) + np.square(x - xc)) / (2 * sigma ** 2))

                        hist, bins = np.histogram(gradient_direction, bins=binsTheta, range=(0, np.pi), weights=gradient_strength)
                        # hist = np.divide(hist, sum(hist))

                        if not np.isnan(np.min(hist)):
                            feature[r, :] += hist / float(len(lesion))
                        else:
                            print('NaNs in RIFT for', scan.uid, 'at radius', str(riftRadii[r]))

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
                    if os.path.isfile(scan.images['t1p']) and os.path.isfile(scan.images['t2w']) and os.path.isfile(scan.images['pdw']) and os.path.isfile(scan.images['flr']):
                        print('Parsing files for', f)
                        mri_list.append(scan)
                        complete_data_subjects += 1
                    else:
                        print('Missing MRI modality: ', f)
                        missing_data_subjects += 1
                else:
                    print('Missing lesion labels: ', f)
                    missing_data_subjects += 1

    print(complete_data_subjects, '/', missing_data_subjects + complete_data_subjects, 'have all modalities and lesion labels')

    mri_list_lesions = []
    for i, scan in enumerate(mri_list):
        scan.lesionList = separate_lesions(scan)
        mri_list_lesions.append(scan)
        print(scan.uid, i+1, '/', len(mri_list)+1)

    return mri_list_lesions


def getICBMContext(scan, images):
    # contextMin = {"csf": -0.001, "wm": -0.001, "gm": -0.001, "pv": -0.001, "lesion": -0.001}
    # contextMax = {'csf': 1.001, 'wm': 1.001, 'gm': 1.001, 'pv': 1.001, 'lesion': 0.348}
    #
    # numBins = 4

    wm_tracts = ['Anterior_Segment', 'Arcuate', 'Cingulum', 'Cortico_Ponto_Cerebellum', 'Cortico_Spinal',
                 'Inferior_Cerebellar_Pedunculus', 'Inferior_Longitudinal_Fasciculus',
                 'Inferior_Occipito_Frontal_Fasciculus', 'Long_Segment', 'Optic_Radiations', 'Posterior_Segment',
                 'Superior_Cerebelar_Pedunculus', 'Uncinate', 'Anterior_Commissure', 'Corpus_Callosum', 'Fornix', 'Internal_Capsule']

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
                
            # contextHist = np.histogram(context, numBins, (contextMin[tissue], contextMax[tissue]))
            # contextHist = contextHist[0] / np.sum(contextHist[0], dtype='float')
            #
            # if np.isnan(contextHist).any():
            #     contextHist = np.zeros(numBins)
            #     contextHist[0] = 1

            saveDocument[tissue] = [np.mean(context), np.var(context)]

        pickle.dump(saveDocument, open(scan.features_dir + 'context_' + str(l) + '.pkl', "wb"))


def getLBPFeatures(scan, images):
    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)

        if len(lesion) > 100:
            size = 'large'
        elif len(lesion) > 25:
            size = 'medium'
        elif len(lesion) > 10:
            size = 'small'
        elif len(lesion) > 2:
            size = 'tiny'
        else:
            continue
            
        if size == 'large' or size == 'medium':
            feature = np.zeros((len(lbpRadii), 9))
        elif size == 'small' or size == 'tiny':
            feature = np.zeros((len(lbpRadii), 9))
        else:
            continue
            
        for j, mod in enumerate(modalities):
            saveDocument[mod] = {}
            
            for r, radius in enumerate(lbpRadii):
                feature[r, ...] = uniformLBP(images[mod], lesion, radius)
            saveDocument[mod] = feature

        pickle.dump(saveDocument, open(scan.features_dir + 'lbp_' + str(l) + '.pkl', "wb"))


def getIntensityFeatures(scan, images):
    # intensityMin = {"t1p": 32.0, "t2w": 10.0, "flr": 33.0, "pdw": 49.0}
    # intensityMax = {'t1p': 1025.0, 't2w': 1000.0, 'flr': 1016.0, 'pdw': 1018.0}

    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)
        
        histBins = 4
        
        for m in modalities:
            intensities = []
            for point in lesion:
                intensities.append(images[m][point[0], point[1], point[2]])
            
            # intensityHist = np.histogram(intensities, histBins, (intensityMin[m], intensityMax[m]))
            # intensityHist = intensityHist[0] / np.sum(intensityHist[0], dtype='float')
            #
            # if np.isnan(intensityHist).any():
            #     intensityHist = np.zeros((histBins))
            #     intensityHist[0] = 1

            saveDocument[m] = [np.mean(intensities), np.var(intensities)]
        
        pickle.dump(saveDocument, open(scan.features_dir + 'intensity_' + str(l) + '.pkl', "wb"))


def getFeaturesOfList(mri_list, riftRegions):
    for i, scan in enumerate(mri_list):
        images = {}
        for j, m in enumerate(modalities):
            images[m] = nib.load(scan.images[m]).get_data()
        
        print('Patient:', scan.uid, i+1, '/', len(mri_list)+1)
        startTime = time.time()

        if doContext:
            getICBMContext(scan, images)

        if doLBP:
            getLBPFeatures(scan, images)
        
        if doRIFT:
            getRIFTFeatures2D(scan, riftRegions, images)
        
        if doIntensity:
            getIntensityFeatures(scan, images)
            
        elapsed = time.time() - startTime
        print(elapsed, "seconds")


def chunks(l, n):
    shuffle(l)
    for i in range(0, len(l), n):
        yield l[i:i+n]


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

    riftRegions = generateRIFTRegions2D(riftRadii)
    print('extracting imaging ')
    getFeaturesOfList(mri_list, riftRegions)

    print('writing clinical outputs...')
    write_clinical_outputs(mri_list)
    print('Done')
    
    endTime = time.time()
    
    elapsed = endTime - startTime
    print("Total time elapsed:", elapsed/3600, 'hours', elapsed/60, 'minutes')


if __name__ == "__main__":
    main()
