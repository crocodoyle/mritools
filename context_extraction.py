import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import multiprocessing as mp

import vtk

from mri import mri
#from scipy import stats

#from scipy.ndimage.filters import convolve
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.interpolation import map_coordinates

import cPickle as pickle

import subprocess
import time, sys

from random import shuffle
from pymongo import MongoClient
from bson.binary import Binary
import pymongo


from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_hit_or_miss

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import center_of_mass

from scipy.spatial import Voronoi

import matplotlib.pyplot as plt
import transformations as t

import bitstring 

icbmRoot = '/usr/local/data/adoyle/trials/quarantine/common/models/icbm_avg_152_'
data_dir = '/usr/local/data/adoyle/trials/MS-LAQ-302-STX/'
lesion_atlas = '/usr/local/data/adoyle/trials/quarantine/common/models/icbm_avg_3714_t2les.mnc.gz'


threads = 8
recompute = False
reconstruct = False

doGabors = False
doLBP = True
doContext = False
doRIFT = False

reload_list = False
multithreaded = False


modalities = ['t1p', 't2w', 'pdw', 'flr']


#blizzard ip
dbIP = '132.206.73.115'
dbPort = 27017


lbpBinsTheta = 6
lbpBinsPhi = 4


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
        
def getLesionSizes(mri_list):
    numLesions = 0
    
    lesionSizes = []
    brainUids = []
    lesionCentroids = []
    
    for i, scan in enumerate(mri_list):
        for j, lesion in enumerate(scan.lesionList):
            numLesions += 1
            lesionSizes.append(len(lesion))
            brainUids.append(scan.uid)
                
            x, y, z = [int(np.mean(x)) for x in zip(*lesion)]
            lesionCentroids.append((x, y, z))
                
    return numLesions, lesionSizes, lesionCentroids, brainUids


def getBoundingBox(mri_list):
    lesTypes = ['tiny', 'small', 'medium', 'large']    
    boundingBoxes = {}    

    xMax = {}
    yMax = {}
    zMax = {}
    for lesType in lesTypes:
        xMax[lesType] = 0
        yMax[lesType] = 0
        zMax[lesType] = 0
    
    for scan in mri_list:
        for les in scan.lesionList:
            if (len(les) > 2) and (len(les) < 11):
                lesType = 'tiny'
            if (len(les) > 10) and (len(les) < 26):
                lesType = 'small'
            if (len(les) > 25) and (len(les) < 101):
                lesType = 'medium'
            if (len(les) > 100):
                lesType = 'large'
            if (len(les)) < 3:
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
        
    print 'boundingBoxes: ', boundingBoxes
    return boundingBoxes
    

def generateGabors():
    rotTheta = np.linspace(0,180, num=4, endpoint=False)
    rotPhi = np.linspace(0,90, num=2, endpoint=False)
    
    windowSizes = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]
    
    sigmaX = [0.2, 1.0]
    sigmaY = [0.6, 1.0]
    
    frequencies = [2, 4, 8]
    
    gabors = []
        
    for rot in rotTheta:
        for rot2 in rotPhi:
            for width in windowSizes:
                for freq in frequencies:
                    for sigX in sigmaX:
                        for sigY in sigmaY:
                            newGab = gabor(rot, rot2, width, freq, sigX, sigY, 1)
                            for g in gabors:
                                if np.array_equal(newGab,g):
                                    print 'already have that one'
                            gabors.append(newGab)
                    
    return gabors


def gabor(rotationTheta, rotationPhi, width, freq, sigX, sigY, sigZ):

    gabor_range = range(-int(width)/2, int(width)/2+1)
    
    sigX *= float(width/4)
    sigY *= float(width/4)
    sigZ *= float(width/4) 
    
    gaus = np.zeros((len(gabor_range),len(gabor_range),len(gabor_range)))
    sine = np.zeros((len(gabor_range),len(gabor_range),len(gabor_range)))
        
    for i, m in enumerate(gabor_range):
        for j, n in enumerate(gabor_range):
            for k, o in enumerate(gabor_range):
                gaus[i,j,k] = (1/((2*np.pi**1.5)*sigX*sigY*sigZ))*np.exp(-((m/sigX)**2 + (n/sigY)**2 + (o/sigZ)**2)/2)
#                sine[i,j,k] = np.real(S*np.exp(2j*np.pi*(u*m + v*n + w*o)))
                sine[i,j,k] = np.real(np.exp(2j*np.pi*freq*np.sqrt(m**2 + 1 + 1)))
    
    gab = np.multiply(gaus, sine)
    gab = np.divide(gab, np.max(gab))
        
    gab = rotate(gab, rotationTheta, axes=(1,2), reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
    gab = rotate(gab, rotationPhi, axes=(0,2), reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=True)    
    gab = zoom(gab, [1, 1, 1.0/3.0])
    
    return gab

def lbp(image, lesionPoints, radius):
    lbpPatterns = np.zeros((len(lesionPoints), lbpBinsTheta*lbpBinsPhi), dtype='float')
    
    for l, [x, y, z] in enumerate(lesionPoints):
        sampleAt = lbpSphere(radius, [x,y,z], lbpBinsTheta, lbpBinsPhi)        
        sampledPoints = map_coordinates(image, sampleAt, order=1)
        
        for m, sampledValue in enumerate(sampledPoints):
            if image[x,y,z] > sampledValue:
                lbpPatterns[l, m] = 1
        
    lbpFeature = np.zeros((lbpBinsTheta*lbpBinsPhi), dtype='float')
    
    for l in range(np.shape(lesionPoints)[0]):
        lbpFeature += lbpPatterns[l,:] / float(len(lesionPoints))

    return lbpFeature
    
def regionLBP(image, regionPoints, radius):
    lbp = bitstring.BitArray('0b000000000000000000000000')
    allLBP = []
    
    for i, [x, y, z] in enumerate(regionPoints):
        sampleAt = lbpSphere(radius, [x,y,z], lbpBinsTheta, lbpBinsPhi)
        sampledPoints = map_coordinates(image, sampleAt, order=1)
        
        for m, sampledValue in enumerate(sampledPoints):
            if image[x,y,z] > sampledValue:
                lbp[m] = True

        allLBP.append(lbp.uint)


    lbpHist = np.zeros((2**(lbpBinsTheta*lbpBinsPhi)))
    for val in allLBP:
        lbpHist[val] += 1
    
    return lbpHist
    

#generates a lookup table of points to sample for RIFT function
def generateRIFTRegions(radii):
    pointLists = []
    
    for r in range(len(radii)):
        pointLists.append([])
        
    for x in range(-np.max(radii), np.max(radii)):
        for y in range(-np.max(radii), np.max(radii)):
            for z in range(-np.max(radii), np.max(radii)):
                distance = np.sqrt((x)**2 + y**2 + (z*(1.0/3.0))**2)
                
                if distance < radii[0]:
                    pointLists[0].append([x, y, z])
                if distance >= radii[0] and distance < radii[1]:
                    pointLists[1].append([x, y, z])
                if distance >= radii[1] and distance < radii[2]:
                    pointLists[2].append([x, y, z])
    
    return pointLists


def getRIFTFeatures(scan, riftRegions):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    radii = [2, 4, 6]
    
    numBinsTheta = 8
    numBinsPhi = 4    
    
    binsTheta = np.linspace(0, 2*np.pi, num=numBinsTheta+1, endpoint=True)
    binsPhi = np.linspace(0, 2*np.pi, num=numBinsPhi+1, endpoint=True)
    
    img = {}
    grad_x = {}
    grad_y = {}
    grad_z = {}

    mag = {}
    theta = {}
    phi = {}
    
    for mod in modalities:
        img[mod] = nib.load(scan.images[mod]).get_data()

        grad_x[mod], grad_y[mod], grad_z[mod] = np.gradient(img[mod])
    
        mag[mod] = np.sqrt(np.square(grad_x[mod]) + np.square(grad_y[mod]) + np.square(grad_z[mod]))
        theta[mod] = np.arctan2(grad_y[mod], grad_x[mod])
        phi[mod] = np.arctan2(grad_x[mod], grad_z[mod])

    for l, lesion in enumerate(scan.lesionList):        
        if db['rift'].find({'_id': scan.uid + '_' + str(l), 't1p':{"$exists":True}, 't2w':{"$exists":True}, 'pdw':{"$exists":True}, 'flr':{"$exists":True}}).count() > 0:
            continue
        
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)        
        
        for mod in modalities:
            feature = np.zeros((len(lesion), len(radii), numBinsTheta*numBinsPhi))
            
            for pIndex, point in enumerate(lesion):
                xc, yc, zc = point

                for r, region in enumerate(riftRegions):
                    gradientData = np.zeros((len(region), 3))

                    for p, evalPoint in enumerate(region):
                        x = xc + evalPoint[0]
                        y = yc + evalPoint[1]
                        z = zc + evalPoint[2]
                        
                        relTheta = np.arctan2((y - yc), (x - xc))
                        relPhi = np.arctan2((x - xc), (z - zc))
                        
                        outwardTheta = (theta[mod][x,y,z] - relTheta + 2*np.pi)%(2*np.pi)
                        outwardPhi = (phi[mod][x,y,z] - relPhi + 2*np.pi)%(2*np.pi)

                        gradientData[p,:] = [outwardTheta, outwardPhi, mag[mod][x,y,z]]
                    
#                    print 'theta', np.max(gradientData[:,0]) / np.pi, np.min(gradientData[:, 0]) / np.pi
#                    print 'phi', np.max(gradientData[:,1]) / np.pi, np.min(gradientData[:,1]) / np.pi
#                    
                    H = np.histogram2d(gradientData[:, 0], gradientData[:,1], [binsTheta, binsPhi], weights=gradientData[:,2])
                    feature[pIndex, r, :] = np.reshape(H[0], (numBinsTheta*numBinsPhi))                    
                    
            saveDocument[mod] = Binary(pickle.dumps(np.mean(feature, axis=0)))
    
        for i in range(30):
            try:
                db['rift'].update_one({'_id' : scan.uid + '_' + str(l)}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect:
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['MSLAQ']
                time.sleep(2*i)
        
def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        if j: y += int(j)<<i
    return y
    
    
def loadMRIList():
    total = 0    
    
    mri_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('_m0_t1p.mnc.gz'):
                scan = mri(f)

                if os.path.isfile(scan.lesions) and os.path.isfile(scan.images['t1p']) and os.path.isfile(scan.images['t2w']) and  os.path.isfile(scan.images['pdw']) and os.path.isfile(scan.images['flr']):
                    scan.separateLesions()
                    mri_list.append(scan)
                    total += 1
                    
                    print scan.images['t1p']
    return mri_list

def lbpSphere(radius, centre, pointsTheta, pointsPhi):
    sampleAt = []
    theta = np.linspace(0, np.pi, num=pointsTheta, endpoint=False)
    phi = np.linspace(0, np.pi*2, num=pointsPhi, endpoint=False)
    
    for th in theta:
        for p in phi:
            x = radius*np.sin(p)*np.cos(th) + centre[0]
            y = radius*np.sin(p)*np.sin(th) + centre[1]
            z = radius*np.cos(p)*(1.0/3.0) + centre[2]
            
            sampleAt.append((x,y,z))
    
    return np.asarray(np.transpose(sampleAt))


def getICBMContext(scan): 
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
        
    tissueContext = {}
        
    img = {}
    
#    print 'mincresample -transformation', scan.lesionPriorXfm, '-invert_transformation', lesion_atlas, scan.priors['lesion'], '-like', scan.priors['wm']

#    if not os.path.exists(scan.lesionPriorXfm):
#        print 'transform file missing'
#    if not os.path.exists(lesion_atlas):
#        print 'lesion atlas doesnt exist'
    
#    return_value = subprocess.call(['mincresample', '-transformation', scan.lesionPriorXfm, '-invert_transformation', lesion_atlas, scan.priors['lesion'], '-like', scan.priors['wm']])

#    print 'mincresample return value:', return_value
        
#    if not os.path.exists(scan.priors['lesion']):
#    subprocess.call(['mv', scan.priors['lesion'], scan.priors['lesion'][0:-3]])
#    print 'mv', scan.priors['lesion'], scan.priors['lesion'][0:-3]
#    subprocess.call(['mincresample', '-transformation', scan.lesionPriorXfm, '-invert_transformation', lesion_atlas, scan.priors['lesion'], '-like', scan.priors['wm']])
#    subprocess.call(['gzip', '-f', scan.priors['lesion'][0:-3]])
    
    
    for tissue in scan.tissues:
        filename = scan.priors[tissue]
        try:
            img[tissue] = nib.load(filename).get_data()
        except Exception as e:
            print 'Error for tissue', tissue, ':', e

    for l, lesion in enumerate(scan.lesionList):
        
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)
            
        for tissue in scan.tissues:
            try:
                tissueContext[tissue] = []

                for p in lesion:
                    tissueContext[tissue].append(img[tissue][p[0], p[1], p[2]])
                
                saveDocument[tissue] = Binary(pickle.dumps(np.mean(tissueContext[tissue]), protocol=2))
            except:
#                print 'Couldnt load context for', tissue
                pass
        
        for i in range(30):
            try:
                db['context'].update_one({'_id' : scan.uid + '_' + str(l)}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect:
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['MSLAQ']
                time.sleep(2*i)
        

def getLBPRegions(box, centroid):
    lbpRegions = [[],[],[],[],[],[],[],[],[]]
    regionMasks = [[],[],[],[],[],[],[],[],[]]

    bigBox = np.zeros((box[0]*box[1]*box[2], 3))
    
    index = 0
    for z in range(centroid[0] - box[0]/2, centroid[0] + box[0]/2):
        for y in range(centroid[1] - box[1]/2, centroid[1] + box[1]/2):
            for x in range(centroid[2] - box[2]/2, centroid[2] + box[2]/2):
                bigBox[index, ...] = z, y, x
                index += 1

    regionMasks[0] = (bigBox[:, 0] < centroid[0] + 1) & (bigBox[:, 1] < centroid[1] + 1) & (bigBox[:,2] < centroid[2] + 1)
    regionMasks[1] = (bigBox[:, 0] > centroid[0]) & (bigBox[:, 1] < centroid[1] + 1) & (bigBox[:,2] < centroid[2] + 1)
    regionMasks[2] = (bigBox[:, 0] < centroid[0] + 1) & (bigBox[:, 1] > centroid[1]) & (bigBox[:,2] < centroid[2] + 1)
    regionMasks[3] = (bigBox[:, 0] < centroid[0] + 1) & (bigBox[:, 1] < centroid[1] + 1) & (bigBox[:,2] > centroid[2])
    regionMasks[4] = (bigBox[:, 0] > centroid[0]) & (bigBox[:, 1] > centroid[1]) & (bigBox[:,2] < centroid[2] + 1)
    regionMasks[5] = (bigBox[:, 0] > centroid[0]) & (bigBox[:, 1] < centroid[1] + 1) & (bigBox[:,2] > centroid[2])
    regionMasks[6] = (bigBox[:, 0] < centroid[0] + 1) & (bigBox[:, 1] > centroid[1]) & (bigBox[:,2] > centroid[2])
    regionMasks[7] = (bigBox[:, 0] > centroid[0]) & (bigBox[:, 1] > centroid[1]) & (bigBox[:,2] > centroid[2])

    for i in range(8):
        lbpRegions[i] = bigBox[regionMasks[i]]

    return lbpRegions



def getLBPFeatures(scan, boundingBoxes):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    imageStartTime = time.time()

    images = {}

    lbpRadii = [1, 2, 3]
    
    
    for j, mod in enumerate(modalities):
        print scan.uid, mod
        images[mod] = nib.load(scan.images[mod]).get_data()
        print np.shape(images[mod])

    for l, lesionPoints in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)
        
        #centroid
        centroidZ, centroidY, centroidX = [int(np.mean(x)) for x in zip(*lesionPoints)]

        
        if (len(lesionPoints) > 2) & (len(lesionPoints) < 11):
            box = boundingBoxes['tiny']
        if (len(lesionPoints) > 10) & (len(lesionPoints) < 26):
            box = boundingBoxes['small']
        if (len(lesionPoints) > 25) & (len(lesionPoints) < 101):
            box = boundingBoxes['medium']
        if (len(lesionPoints) > 100):
            box = boundingBoxes['large']
        if (len(lesionPoints) < 3):
            continue
        
        regions = getLBPRegions(box, [centroidZ, centroidY, centroidX])
        
        for j, mod in enumerate(modalities):
            saveDocument[mod] = {}
            
            for r, radius in enumerate(lbpRadii):
                
                lbpHist = np.zeros((len(regions), 2**(lbpBinsTheta*lbpBinsPhi)))
                for i, reg in enumerate(regions):
                    lbpHist[i, :] = regionLBP(images[mod], reg, radius)
                    
                lbpVector = np.hstack((lbpHist[0], lbpHist[1], lbpHist[2], lbpHist[3], lbpHist[4], lbpHist[5], lbpHist[6], lbpHist[7]))
                saveDocument[mod][str(radius)] = Binary(pickle.dumps(lbpVector, protocol=2))
     
        for i in range(30):
            try:
                db['lbp'].update_one({'_id' : scan.uid + '_' + str(l)}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect:
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['MSLAQ']
                time.sleep(2*i)
                
    imageEndTime = time.time()
    elapsed = imageEndTime - imageStartTime
    print elapsed/(60), "minutes", elapsed%60, "seconds"

def getGaborFeatures(scan, gabors, nBest):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    imageStartTime = time.time()
    
    modalities = ['t1p', 't2w', 'pdw', 'flr']
    rotTheta = np.linspace(0,180, num=4, endpoint=False)
    rotPhi = np.linspace(0,90, num=2, endpoint=False)

    gaborWidth = [1, 2.5, 4.5]
    gaborSpacing = [0.01, 0.2, 2]
    
    images = {}

    for j, m in enumerate(modalities):
        images[m] = nib.load(scan.images[m]).get_data()
    
    for l, lesionPoints in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)
        
        if db['gabor'].find({'_id': scan.uid + '_' + str(l), 'pdw': {"$exists" : True}, 't2w': {"$exists" : True}, 't1p':{"$exists" : True}, 'flr':{"$exists" :True} }).count() > 0:
            continue
        
        for j, m in enumerate(modalities):                
            gaborResponses = np.zeros((len(lesionPoints), len(rotTheta), len(rotPhi), len(gaborWidth), len(gaborSpacing)))
            for i, [x, y, z] in enumerate(lesionPoints):
                for r, rot in enumerate(rotTheta):
                    for r2, rot2 in enumerate(rotPhi):
                        for w, width in enumerate(gaborWidth):
                            for o, spacing in enumerate(gaborSpacing):
                                gaborResponses[i, r, r2, w, o] = np.mean(np.multiply(images[m][x-5:x+6,y-5:y+6,z-1:z+2], gabors[r, r2, w, o, :, :, :]))
            
            gaborHists = np.zeros((4, 2, 3))
            
            for w, width in enumerate(gaborWidth):
                for n in range(nBest):
                    index = np.argmax(gaborResponses[:, :, :, w, :])
                    index = np.unravel_index(index, np.shape(gaborResponses[:, :, :, w, :]))
                                        
                    gaborHists[index[1], index[2], index[3]] += (1.0 / (len(lesionPoints)*nBest))
                    gaborResponses[index[0], index[1], index[2], w, index[3]] = 0
          
            saveDocument[m] = Binary(pickle.dumps(gaborHists, protocol=2))
        
        for i in range(30):
            try:
                db['gabor'].update_one({'_id' : scan.uid + '_' + str(l)}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect:
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['MSLAQ']
                time.sleep(2*i)

    imageEndTime = time.time()
    elapsed = imageEndTime - imageStartTime
    print elapsed/(60), "minutes"


def getFeaturesOfList(mri_list, gabors, riftRegions, boundingBoxes):
    
    for i, scan in enumerate(mri_list):
        print scan.uid, i, '/', len(mri_list)
        if doContext:
            getICBMContext(scan)

        if doGabors:
            getGaborFeatures(scan, gabors, 10)

        if doLBP:
            getLBPFeatures(scan, boundingBoxes)
        
        if doRIFT:
            getRIFTFeatures(scan, riftRegions)


def chunks(l, n):
    shuffle(l)
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
    

def main():
    startTime = time.time()

    print 'Loading MRI file list...'
    
    if reload_list:
        print 'Reloading MRI file list from NeuroRX...'
        for root, dirs, filenames in os.walk(data_dir):
            if len(filenames) == 0:
                #sshfs adoyle@iron7.bic.mni.mcgill.ca:/trials/ -p 22101 /usr/local/data/adoyle/trials/
                subprocess.call(['sshfs', 'adoyle@iron7.bic.mni.mcgill.ca:/trials/', '-p', '22101', '/usr/local/data/adoyle/trials/'])
        mri_list = loadMRIList()
        outfile = open('/usr/local/data/adoyle/mri_list.pkl', 'wb')
        pickle.dump(mri_list, outfile)
        outfile.close()
        print 'Cached MRI file listing'
    else:
        infile = open('/usr/local/data/adoyle/mri_list.pkl', 'rb')
        mri_list = pickle.load(infile)
        infile.close()
    
    print 'MRI list loaded'
    
    mri_list = convertToNifti(mri_list)
#    mri_list = gzipNiftiFiles(mri_list)
    numLesions, lesionSizes, lesionCentroids, brainUids = getLesionSizes(mri_list)
    
    boundingBoxes = getBoundingBox(mri_list)    
    
    gabors = generateGabors()    
    riftRegions = generateRIFTRegions([2,4,6])
    
    
    if multithreaded:
        chunkSize = (len(mri_list) / threads) + 1
        
        procs = []        
        for i, sublist in enumerate(chunks(mri_list, chunkSize)):
            print 'Starting process', i, 'with', len(sublist), 'images to process'
            worker = mp.Process(target=getFeaturesOfList, args=(sublist, gabors, riftRegions, boundingBoxes))
            worker.start()
            procs.append(worker)
        
        print 'started processes'    
        
        for proc in procs:
            print 'Waiting...'
            proc.join()
    else:  
        getFeaturesOfList(mri_list, gabors, riftRegions, boundingBoxes)

    print 'Done'
    
    endTime = time.time()
    
    elapsed = endTime - startTime
    print "Total time elapsed:", elapsed/3600, 'hours', elapsed/60, 'minutes'

def displayGabors():
    gabors = generateGabors()
    
    print len(gabors)
    
    for g in gabors:
        plt.imshow(g[int(np.shape(g)[0]/2), :, :], interpolation='nearest')
        plt.show()
                     
    plt.show()

def displayBrains():
    infile = open('/usr/local/data/adoyle/mri_list.pkl', 'rb')
    mri_list = pickle.load(infile)
    infile.close()
    
    lesionSizes = []
    lesionsInBrain = []
    
    for scan in mri_list:
        lesionsInBrain.append(len(scan.lesionList))
        
        lesionBins = [0, 0, 0]
        for lesion in scan.lesionList:
            if len(lesion) > 2:
                
                if len(lesion) <= 25:
                    lesionBins[0] +=1
                elif len(lesion) <= 1500:
                    lesionBins[1] += 1
                else:
                    lesionBins[2] += 1
                
                lesionSizes.append(len(lesion))
            
    print len(lesionSizes)
    plt.bar(range(len(lesionBins)), lesionBins)
    plt.title('Distribution of Lesion Sizes')
    xticks = ['<25', '<1500', '1500+']
    plt.xlabel('Lesion Size (voxels)')
    plt.xticks(range(len(xticks)), xticks)
    plt.show()
    
    plt.hist(lesionsInBrain)
    plt.title('Distribution of Lesions per Brain')
    plt.xlabel('Lesions in Brain')
    plt.show()
    
    
    for scan in mri_list[0:5]:
        mri_data = nib.load(scan.images['flr']).get_data()
        plt.imshow(mri_data[30,:, :], cmap = cm.Greys_r)
        plt.axis('off')        
        plt.show()


def hitOrMissThinning(lesion):
    img = np.zeros((60, 256, 256), dtype='bool')
    
    for point in lesion:
        img[point[0], point[1], point[2]] = 1
        
    elem = np.zeros((3, 3, 3), dtype='bool')
    elem[1,1,1] = 1
    elem[1,0,:] = 1
    
    elemConv = np.zeros((3, 3, 3), dtype='bool')
    elemConv[1,2,:] = 1
    
    
    elem2 = np.zeros((3, 3, 3), dtype='bool')
    elem2[1,1,1] = 1
    elem2[1,0,1] = 1
    elem2[1,1,0] = 1
    elem2[2,1,1] = 1
    elem2[0,1,1] = 1
    
    elem2Conv = np.zeros((3, 3, 3), dtype='bool')
    elem2Conv[1,1,2] = 1
    elem2Conv[1,2,2] = 1
    elem2Conv[1,2,1] = 1
    
#    elem2Conv[0,1,1] = 1
#    elem2Conv[2,1,1] = 1    
#    
#    
#    elem2Conv[2,1,2] = 1
#    elem2Conv[0,1,2] = 1
#    elem2Conv[2,2,1] = 1
#    elem2Conv[0,2,1] = 1
    
    
    rotationsTheta = [0, np.pi/2, np.pi, 3*np.pi/2]
    rotationsPhi = [np.pi/2]
    
    iteration = 1
    
    origImg = np.zeros(np.shape(img))
    while not np.sum(origImg) == np.sum(img):
          
        for r in rotationsTheta:
            R = t.rotation_matrix(r, (1, 0, 0), point=(1,1,1))[0:3, 0:3]
            
            e1 = R*elem
            e1Conv = R*elemConv
            
            e2 = R*elem2
            e2Conv = R*elem2Conv
            
            remove = binary_hit_or_miss(img, e1, e1Conv)
            img = img - remove
            remove = binary_hit_or_miss(img, e2, e2Conv)
            img = img - remove
            
#            if iteration % 3 == 0:
#                eZ1 = R*elem
#                eZ1Conv = R*elemConv
#                
#                eZ2 = R*elem2
#                eZ2Conv = R*elem2Conv
                
            for r in rotationsPhi:
                    
                R = t.rotation_matrix(r, (0, 1, 0), point=(1,1,1))[0:3, 0:3]
                e1 = R*e1
                e1Conv = R*e1Conv
                
                e2 = R*e2
                e2Conv = R*e2Conv
                    
                remove = binary_hit_or_miss(img, e1, e1Conv)
                img = img - remove
                remove = binary_hit_or_miss(img, e2, e2Conv)
                img = img - remove
                
        iteration += 1
        origImg = img
    
    print np.sum(img), '/', len(lesion)
    return img

def voroSkeleton(lesion):
    
    skeleton = []

    vor = Voronoi(lesion)
        
    for region in vor.regions:
        if region.all() >= 0:
            for pointIndex in region:
                skeleton.append(vor.vertices[pointIndex])

def getLesionSkeleton(scan):
#        flair = nib.load(self.images['t1p']).get_data()

    struct = np.zeros((3, 3, 3))
    struct[1, 1, 1] = 1
    struct[2, 1, 1] = 1
    struct[1, 2, 1] = 1
    struct[1, 1, 2] = 1
    struct[0, 1, 1] = 1
    struct[1, 0, 1] = 1
    struct[1, 1, 0] = 1

    structNoZ = struct
    structNoZ[0, 1, 1] = 0
    structNoZ[2, 1, 1] = 0

    for lesion in scan.lesionList:
        hitMissSkele = hitOrMissThinning(lesion)
#                vorImg = np.zeros((60, 256, 256))
#                vor = Voronoi(lesion)                
#                
#                goodVertices = []
#                
     
#                for ridge_index in vor.ridge_vertices:
#                    if np.all(ridge_index > 0):
#                        for index in ridge_index:
#                            if vor.vertices[index][0] > 0 and vor.vertices[index][1] > 0 and vor.vertices[index][2] > 0:
#                                point = (int(vor.vertices[index][0]), int(vor.vertices[index][1]), int(vor.vertices[index][2]))
#                                vorImg[point] = 1     
#                                goodVertices.append(point)
#                                
#                
#                for v in vor.vertices:
#                    vorImg[int(v[0]), int(v[1]), int(v[2])] = 1
                
        img = np.zeros((60, 256, 256), dtype='float')
                
        for point in lesion:
            img[point[0], point[1], point[2]] = 1
            
        keepGoing = True
        iteration = 0
        prevImg = img
        while keepGoing:
            keepGoing = False
            if iteration%3 == 0:
                newImg = binary_erosion(prevImg, structure=struct)
            else:
                newImg = binary_erosion(prevImg, structure=structNoZ)
                
            skeletonPoints = np.transpose(np.nonzero(newImg))
            
            if len(skeletonPoints) < 1:
                newImg = prevImg
                keepGoing = False
            else:
                prevImg = newImg
                for point in skeletonPoints:
                    connectedPoints = 0
                    for point2 in skeletonPoints:
                        if np.abs(point[0] - point2[0]) <= 1 and np.abs(point[1] - point2[1]) <= 1 and np.abs(point[2] - point[2]) <= 1:
                            connectedPoints += 1
                    if connectedPoints > 3:
                        keepGoing = True
                        break
        
        boundaryDistance = distance_transform_edt(img, sampling=[3, 1, 1])
        
        point = center_of_mass(img)
        
        centrePoint = (int(point[0]), int(point[1]), int(point[2]))
        distanceGrad = np.abs(np.gradient(boundaryDistance))
        
        sumGrads = distanceGrad[0] + distanceGrad[1] + distanceGrad[2]
        sumGrads = np.multiply(img, sumGrads)
        
    #           plt.imshow(img[centrePoint[0], centrePoint[1]-5:centrePoint[1]+5, centrePoint[2]-5:centrePoint[2]+5])
    #           plt.show()
        
    #           boundaryDistance = np.multiply(boundaryDistance, img)
        if len(lesion) > 10:

            skeletonPointsThin = np.transpose(np.nonzero(hitMissSkele))
            
            displaySkeleton3D(lesion, skeletonPointsThin, np.transpose(np.nonzero(newImg)))
            
            
            plt.subplot(1, 4, 1)     
            plt.axis('off')
            plt.imshow(newImg[centrePoint[0], centrePoint[1]-10:centrePoint[1]+10, centrePoint[2]-10:centrePoint[2]+10], cmap = plt.cm.gray, interpolation = 'nearest')
            
    #               plt.subplot(1, 6, 1)
    #               plt.axis('off')
    #               plt.imshow(distanceGrad[0][centrePoint[0], centrePoint[1]-5:centrePoint[1]+5, centrePoint[2]-5:centrePoint[2]+5], cmap = plt.cm.gray, interpolation = 'nearest')
    #   #            plt.colorbar()
    #               plt.subplot(1, 6, 2)
    #               plt.axis('off')
    #               plt.imshow(distanceGrad[1][centrePoint[0], centrePoint[1]-5:centrePoint[1]+5, centrePoint[2]-5:centrePoint[2]+5], cmap = plt.cm.gray, interpolation = 'nearest')
    #   #            plt.colorbar()            
    #               plt.subplot(1, 6, 3)
    #               plt.axis('off')
    #               plt.imshow(distanceGrad[2][centrePoint[0], centrePoint[1]-5:centrePoint[1]+5, centrePoint[2]-5:centrePoint[2]+5], cmap = plt.cm.gray, interpolation = 'nearest')
    #            plt.colorbar()
            plt.subplot(1, 4, 2)     
            plt.axis('off')
            plt.imshow(boundaryDistance[centrePoint[0], centrePoint[1]-10:centrePoint[1]+10, centrePoint[2]-10:centrePoint[2]+10], cmap = plt.cm.gray, interpolation = 'nearest')
    #            plt.colorbar()
            plt.subplot(1, 4, 3)     
            plt.axis('off')
            plt.imshow(img[centrePoint[0], centrePoint[1]-10:centrePoint[1]+10, centrePoint[2]-10:centrePoint[2]+10], cmap = plt.cm.gray, interpolation = 'nearest')            
    #            plt.colorbar()
    #               plt.subplot(1, 6, 6)
    #               plt.axis('off')
    #               plt.imshow(sumGrads[centrePoint[0], centrePoint[1]-5:centrePoint[1]+5, centrePoint[2]-5:centrePoint[2]+5], cmap = plt.cm.gray, interpolation = 'nearest')
    #             
            plt.subplot(1, 4, 4)     
            plt.axis('off')
            plt.imshow(hitMissSkele[centrePoint[0], centrePoint[1]-10:centrePoint[1]+10, centrePoint[2]-10:centrePoint[2]+10], cmap = plt.cm.gray, interpolation = 'nearest')            
            
            
            plt.show()
 
def displaySkeleton3D(lesion, skeleton, skeleton2):
    
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()

    points2 = vtk.vtkPoints()
    vertices2 = vtk.vtkCellArray()     

    points3 = vtk.vtkPoints()
    vertices3 = vtk.vtkCellArray()     
    
    
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    Colors2 = vtk.vtkUnsignedCharArray()
    Colors2.SetNumberOfComponents(3)
    Colors2.SetName("Colors2")
    Colors3 = vtk.vtkUnsignedCharArray()
    Colors3.SetNumberOfComponents(3)
    Colors3.SetName("Colors3")
    
    for point in lesion:
        pointId = points.InsertNextPoint(point)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pointId)
        Colors.InsertNextTuple3(255,255,255)

    for point in skeleton:
        pointId = points2.InsertNextPoint(point)
        vertices2.InsertNextCell(1)
        vertices2.InsertCellPoint(pointId)
        Colors2.InsertNextTuple3(0,255,0)
        
    for point in skeleton2:
        pointId = points3.InsertNextPoint(point)
        vertices3.InsertNextCell(1)
        vertices3.InsertCellPoint(pointId)
        Colors3.InsertNextTuple3(255,0,0)
                

    poly = vtk.vtkPolyData()
    poly2 = vtk.vtkPolyData()
    poly3 = vtk.vtkPolyData()

    poly.SetPoints(points)
    poly.SetVerts(vertices)
    poly.GetPointData().SetScalars(Colors)
    poly.Modified()
    poly.Update()


#    delaunay = vtk.vtkDelaunay2D()
#    delaunay.SetInput(poly)
#    delaunay.SetSource(poly)
#    delaunay.SetAlpha(0.5)
#    delaunay.Update()
#    
#    delMapper = vtk.vtkDataSetMapper()
#    delMapper.SetInputConnection(delaunay.GetOutputPort())
#    
#    delActor = vtk.vtkActor()
#    delActor.SetMapper(delMapper)
#    delActor.GetProperty().SetInterpolationToFlat()
#    delActor.GetProperty().SetRepresentationToWireframe()

    poly2.SetPoints(points2)
    poly2.SetVerts(vertices2)
    poly2.GetPointData().SetScalars(Colors2)
    poly2.Modified()
    poly2.Update()
    
    poly3.SetPoints(points3)
    poly3.SetVerts(vertices3)
    poly3.GetPointData().SetScalars(Colors3)
    poly3.Modified()
    poly3.Update()

    
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
     
    renWin.SetSize(500, 500)

    mapper = vtk.vtkPolyDataMapper()
    mapper2 = vtk.vtkPolyDataMapper()
    mapper3 = vtk.vtkPolyDataMapper()
    mapper.SetInput(poly)
    mapper2.SetInput(poly2)
    mapper3.SetInput(poly3)
    
    
    transform1 = vtk.vtkTransform()
    transform1.Translate(0.0, 0.1, 0.0)
    transform2 = vtk.vtkTransform()
    transform2.Translate(0.0, 0.0, 0.1)    
    
    
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    actor.GetProperty().SetPointSize(5)
    
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.SetUserTransform(transform1)
    actor2.GetProperty().SetPointSize(5)

    actor3 = vtk.vtkActor()
    actor3.SetMapper(mapper3)
    actor3.SetUserTransform(transform2)
    actor3.GetProperty().SetPointSize(5)
    
    ren.AddActor(actor)
    ren.AddActor(actor2)
    ren.AddActor(actor3)
#    ren.AddActor(delActor)
    ren.SetBackground(.2, .3, .4)
    
    renWin.Render()
    iren.Start()

           
def displaySkeletons():
    infile = open('/usr/local/data/adoyle/mri_list.pkl', 'rb')
    mri_list = pickle.load(infile)
    infile.close()
    
    
    for scan in mri_list[0:10]:
        getLesionSkeleton(scan)
    

if __name__ == "__main__":
#    infile = open('/usr/local/data/adoyle/mri_list.pkl', 'rb')
#    mri_list = pickle.load(infile)
#    
#    for i, scan in enumerate(mri_list[0:2]):
#        getICBMContext(scan)
#        print i, '/', len(mri_list)


#    displayGabors() 
    main()


#    convertToNifti(mri_list)
#    displaySkeletons()

#    displayBrains()
#    gab = gabor(0, 0, 100, 4.5)
#
#    allvals = np.hstack((gab[:,:,0], gab[:,:,1], gab[:,:,2]))
#
#    plt.hist(allvals.flat, bins=32)
#    plt.show()
#    
#    plt.imshow(gab[:,:,1])
#    plt.colorbar()
#    plt.show()    
#    
#    print np.max(gab)
#    print np.min(gab)