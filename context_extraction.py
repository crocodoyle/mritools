import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import multiprocessing as mp


from mri import mri
#from scipy import stats

#from scipy.ndimage.filters import convolve
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.interpolation import map_coordinates

from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm



import cPickle as pickle

import subprocess
import time, sys

from random import shuffle
from pymongo import MongoClient
from bson.binary import Binary
import pymongo




import skeletons

import matplotlib.pyplot as plt
import transformations as t

import bitstring 

#sys.path.append('/usr/local/data/adoyle/SHTOOLS')
#import pyshtools as shtools



icbmRoot = '/usr/local/data/adoyle/trials/quarantine/common/models/icbm_avg_152_'
data_dir = '/usr/local/data/adoyle/trials/MS-LAQ-302-STX/'
lesion_atlas = '/usr/local/data/adoyle/trials/quarantine/common/models/icbm_avg_3714_t2les.mnc.gz'


threads = 8
recompute = False
reconstruct = False

doShape = False
doGabors = False

doLBP = True
doContext = True
doRIFT = True
doIntensity = True

reload_list = False
multithreaded = False


modalities = ['t1p', 't2w', 'pdw', 'flr']


#blizzard ip
dbIP = '132.206.73.115'
dbPort = 27017

riftRadii = [1,2,3]
lbpRadii = [1,2,3]

lbpBinsTheta = 6
lbpBinsPhi = 4

thinners = skeletons.thinningElements()


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
        uniformPatterns = np.zeros((9))
        
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
#        uniformPatterns = np.zeros((8, 9))
#        quadrants = lesionQuadrants(lesion, downSample=1)
        uniformPatterns = np.zeros((9))
        garbage, skeleton = skeletons.hitOrMissThinning(lesion, thinners)
        
#        for q, quadrant in enumerate(quadrants):
#            for i, [x,y,z] in enumerate(quadrant):
        for i, [x,y,z] in enumerate(skeleton):
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
#                    uniformPatterns[q, ones] += 1.0 / float(len(lesion))
                    uniformPatterns[ones] += 1.0 / float(len(lesion))
                else:
#                    uniformPatterns[q, 8] += 1.0 / float(len(lesion))
                    uniformPatterns[8] += 1.0 / float(len(lesion))
                
    return uniformPatterns
    
    
def simpleLBP(image, lesion, radius):
    numQuadrants = 8
    
    size = ""
    if (len(lesion) > 2) and (len(lesion) < 11):
        size = 'tiny'
    elif (len(lesion) > 10) and (len(lesion) < 26):
        size = 'small'
    elif (len(lesion) > 25) and (len(lesion) < 101):
        size = 'medium'
    elif (len(lesion) > 100):
        size = 'large'
        
    lbp = bitstring.BitArray('0b00000000')

    r = radius    
    lbpHist = []
    
    if size == "tiny" or size == "small":
        lbpHist = np.zeros((2**8), dtype='short')
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
            
            lbpPattern = minimumLBPPattern(lbp)
            
            lbpHist[lbpPattern] += 1

    elif size == "medium" or size == "large":
        lbpHist = np.zeros((numQuadrants, 2**8), dtype='short')
        quadrants = lesionQuadrants(lesion, downSample=4)
        for q, quadrant in enumerate(quadrants):
            for i, [x,y,z] in enumerate(quadrant):
                threshold = image[x,y,z]
                
                lbp.set(image[x-r, y, z] > threshold, 0)
                lbp.set(image[x-r, y+r, z] > threshold, 1)
                lbp.set(image[x, y+r, z] > threshold, 2)
                lbp.set(image[x+r, y+r, z] > threshold, 3)
                lbp.set(image[x+r, y, z] > threshold, 4)
                lbp.set(image[x+r, y-r, z] > threshold, 5)
                lbp.set(image[x, y-r, z] > threshold, 6)
                lbp.set(image[x-r, y-r, z] > threshold, 7)
                
                lbpPattern = minimumLBPPattern(lbp)
                
                lbpHist[q, lbpPattern] += 1
        
    return lbpHist

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

def lbpTop(image, lesionPoints, radius):
    lbpXY = bitstring.BitArray('0b00000000')
    lbpXZ = bitstring.BitArray('0b00000000')
    lbpYZ = bitstring.BitArray('0b00000000')
    
    lbpTopHist = np.zeros((2**8 + 2**8 + 2**8), dtype='float')
    r = radius
    
    for i, [x,y,z] in enumerate(lesionPoints):
        threshold = image[x,y,z]
        
        lbpXY.set(image[x-r, y, z] > threshold, 0)
        lbpXY.set(image[x-r, y+r, z] > threshold, 1)
        lbpXY.set(image[x, y+r, z] > threshold, 2)
        lbpXY.set(image[x+r, y+r, z] > threshold, 3)
        lbpXY.set(image[x+r, y, z] > threshold, 4)
        lbpXY.set(image[x+r, y-r, z] > threshold, 5)
        lbpXY.set(image[x, y-r, z] > threshold, 6)
        lbpXY.set(image[x-r, y-r, z] > threshold, 7)
        
        xy = minimumLBPPattern(lbpXY)
        lbpTopHist[xy] += (1.0 / len(lesionPoints))
        
        lbpXZ.set(image[x-r, y, z] > threshold, 0)
        lbpXZ.set(image[x-r, y, z+r] > threshold, 1)
        lbpXZ.set(image[x, y, z+r] > threshold, 2)
        lbpXZ.set(image[x+r, y, z+r] > threshold, 3)
        lbpXZ.set(image[x+r, y, z] > threshold, 4)
        lbpXZ.set(image[x+r, y, z-r] > threshold, 5)
        lbpXZ.set(image[x, y, z-r] > threshold, 6)
        lbpXZ.set(image[x-r, y, z-r] > threshold, 7)
        xz = minimumLBPPattern(lbpXZ)
        lbpTopHist[2**8 + xz] += (1.0 / len(lesionPoints))
        
        lbpYZ.set(image[x, y-r, z] > threshold, 0)
        lbpYZ.set(image[x, y-r, z+r] > threshold, 1)
        lbpYZ.set(image[x, y, z+r] > threshold, 2)
        lbpYZ.set(image[x, y+r, z+r] > threshold, 3)
        lbpYZ.set(image[x, y+r, z] > threshold, 4)
        lbpYZ.set(image[x, y+r, z-r] > threshold, 5)
        lbpYZ.set(image[x, y, z-r] > threshold, 6)
        lbpYZ.set(image[x, y-r, z-r] > threshold, 7)
        yz = minimumLBPPattern(lbpYZ)
        lbpTopHist[2**8 + 2**8 + yz] += (1.0 / len(lesionPoints))
        
        return lbpTopHist

def minimumLBPPattern(lbpPattern):
    
    minVal = lbpPattern.uint
    for i in range(1,8):
        lbpPattern.ror(1)
        if lbpPattern.uint < minVal:
            minVal = lbpPattern.uint
        
    return minVal
        
    
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


    lbpHist = np.zeros((2**(lbpBinsTheta*lbpBinsPhi)), dtype=np.uint8)
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
                distance = np.sqrt(x**2 + y**2 + (z*(1.0/3.0))**2)
                
                if distance < radii[0]:
                    pointLists[0].append([x, y, z])
                elif distance >= radii[0] and distance < radii[1]:
                    pointLists[1].append([x, y, z])
                elif distance >= radii[1] and distance < radii[2]:
                    pointLists[2].append([x, y, z])
    
    return pointLists
    
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
            if distance > radii[1] and distance <= radii[2]:
                pointLists[2].append([x, y])

    return pointLists
    
def getRIFTFeatures2D(scan, riftRegions, img):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    numBinsTheta = 8
    numQuadrants = 8
    
    sigma = np.sqrt(2)
    
    binsTheta = np.linspace(0, 2*np.pi, num=numBinsTheta+1, endpoint=True)
    
    grad_x = {}
    grad_y = {}
    grad_z = {}

    mag = {}
    theta = {}
    
    for mod in modalities:
        grad_x[mod], grad_y[mod], grad_z[mod] = np.gradient(img[mod])
    
        mag[mod] = np.sqrt(np.square(grad_x[mod]) + np.square(grad_y[mod]))
        theta[mod] = np.arctan2(grad_y[mod], grad_x[mod])
        
    feature = np.zeros((1, len(riftRadii), numQuadrants, numBinsTheta))
    for l, lesion in enumerate(scan.lesionList):
        size = ""
        if (len(lesion) > 2) and (len(lesion) < 11):
            size = 'tiny'
        elif (len(lesion) > 10) and (len(lesion) < 26):
            size = 'small'
        elif (len(lesion) > 25) and (len(lesion) < 101):
            size = 'medium'
#            quadrants = lesionQuadrants(lesion, downSample=1)
        elif (len(lesion) > 100):
            size = 'large'
#            quadrants = lesionQuadrants(lesion, downSample=1)
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
                        gradientData = np.zeros((len(region), 2))
    
                        for p, evalPoint in enumerate(region):
                            x = xc + evalPoint[0]
                            y = yc + evalPoint[1]
                            z = zc
                            
                            relTheta = np.arctan2((y - yc), (x - xc))
                            
                            outwardTheta = (theta[mod][x,y,z] - relTheta + 2*np.pi)%(2*np.pi)
    
                            gaussianWindow = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.square(y-yc) + np.square(x-xc)) / (2 * sigma**2))
                            gradientData[p,:] = [outwardTheta, mag[mod][x,y,z]*gaussianWindow]
                                                        

                        hist, bins = np.histogram(gradientData[:, 0], bins=binsTheta, range=(0, np.pi), weights=gradientData[:,1])
                        hist = np.divide(hist, sum(hist))   
                        if not np.isnan(np.min(hist)):
                            feature[r, :] += hist / float(len(lesion))
  
            elif size == 'medium' or size == 'large':
#                feature = np.zeros((len(riftRadii), numQuadrants, numBinsTheta))
                feature = np.zeros((len(riftRadii), numBinsTheta))

                garbage, skeleton = skeletons.hitOrMissThinning(lesion, thinners)
#                for q, quadrant in enumerate(quadrants):
#                    for pIndex, point in enumerate(quadrant):
                for pIndex, point in enumerate(skeleton):
                        xc, yc, zc = point
                            
                        for r, region in enumerate(riftRegions):
                            gradientData = np.zeros((len(region), 2))
        
                            for p, evalPoint in enumerate(region):
                                x = xc + evalPoint[0]
                                y = yc + evalPoint[1]
                                z = zc
                                
                                relTheta = np.arctan2((y - yc), (x - xc))
                                
                                outwardTheta = (theta[mod][x,y,z] - relTheta + 2*np.pi)%(2*np.pi)
        
                                gaussianWindow = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.square(y-yc) + np.square(x-xc)) / (2 * sigma**2))
                                gradientData[p,:] = [outwardTheta, mag[mod][x,y,z]]
                            
        #                    print 'theta', np.max(gradientData[:,0]) / np.pi, np.min(gradientData[:, 0]) / np.pi
        #                    print 'phi', np.max(gradientData[:,1]) / np.pi, np.min(gradientData[:,1]) / np.pi
      
                            hist, bins = np.histogram(gradientData[:, 0], bins=binsTheta, range=(0, np.pi), weights=gradientData[:,1])
                            hist = np.divide(hist, sum(hist))
                            if not np.isnan(np.min(hist)):
                                feature[r, :] += hist / float(len(skeleton))

            saveDocument[mod] = Binary(pickle.dumps(feature)) 
        for i in range(30):
            try:
                db['rift'].update_one({'_id' : scan.uid + '_' + str(l)}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect:
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['MSLAQ']
                time.sleep(2*i)
                
                
def getRIFTFeatures3D(scan, riftRegions, img):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    
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
            feature = np.zeros((len(lesion), len(riftRadii), numBinsTheta*numBinsPhi))
            
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
    
def loadMRIList():
    total = 0    
    
    mri_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if total > 3:
                break
            if f.endswith('_m0_t1p.mnc.gz'):
                scan = mri(f)
                
                if os.path.isfile(scan.lesions) and os.path.isfile(scan.images['t1p']) and os.path.isfile(scan.images['t2w']) and  os.path.isfile(scan.images['pdw']) and os.path.isfile(scan.images['flr']):
                    scan.separateLesions()
                    mri_list.append(scan)
                    total += 1
                    
                    print total, '/', len(filenames)
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


def getICBMContext(scan, images): 
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
                
    contextMin = {"csf": -0.001, "wm": -0.001, "gm": -0.001, "pv": -0.001, "lesion": -0.001}
    contextMax = {'csf': 1.001, 'wm': 1.001, 'gm': 1.001, 'pv': 1.001, 'lesion': 0.348}
    
    numBins = 4
    
    for tissue in scan.tissues:
        filename = scan.priors[tissue]
        try:
            images[tissue] = nib.load(filename).get_data()
        except Exception as e:
            print 'Error for tissue', tissue, ':', e

    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)
            
        for tissue in scan.tissues:
            context = []

            for p in lesion:
                context.append(images[tissue][p[0], p[1], p[2]])
                
            contextHist = np.histogram(context, numBins, (contextMin[tissue], contextMax[tissue]))
            contextHist = contextHist[0] / np.sum(contextHist[0], dtype='float')
            
            if np.isnan(contextHist).any():
                contextHist = np.zeros((numBins))
                contextHist[0] = 1
        
            filteredContextHist = gaussian_filter(contextHist, float(numBins)/30.0)
#            print 'context', tissue, contextHist
#            print 'context', tissue, filteredContextHist
##            sys.stdout.flush()
            saveDocument[tissue] = Binary(pickle.dumps([np.mean(context), np.var(context)], protocol=2))

        
        for i in range(30):
            try:
                db['context'].update_one({'_id' : scan.uid + '_' + str(l)}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect:
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['MSLAQ']
                time.sleep(2*i)
        

#def lesionQuadrants(box, centroid, downSample=1):
#    quadrantPoints = [[],[],[],[],[],[],[],[],[]]
#    regionMasks = [[],[],[],[],[],[],[],[],[]]
#
#    bigBox = np.zeros((box[0]*box[1]*box[2], 3), dtype=np.uint8)
#    
#    index = 0
#    for z in range(centroid[2] - box[2]/2, centroid[2] + box[2]/2):
#        for i, x in enumerate(range(centroid[0] - box[0]/2, centroid[0] + box[0]/2)):
#            for j, y in enumerate(range(centroid[1] - box[1]/2, centroid[1] + box[1]/2)):
#                bigBox[index, ...] = x, y, z
#                index += 1
#                    
#    for i in range(box[0]*box[1]*box[2], index, -1):
#        bigBox = np.delete(bigBox, (i-1), axis=0)
#
#    regionMasks[0] = (bigBox[:, 0] < centroid[0] + 1) & (bigBox[:, 1] < centroid[1] + 1) & (bigBox[:,2] < centroid[2] + 1)
#    regionMasks[1] = (bigBox[:, 0] > centroid[0]) & (bigBox[:, 1] < centroid[1] + 1) & (bigBox[:,2] < centroid[2] + 1)
#    regionMasks[2] = (bigBox[:, 0] < centroid[0] + 1) & (bigBox[:, 1] > centroid[1]) & (bigBox[:,2] < centroid[2] + 1)
#    regionMasks[3] = (bigBox[:, 0] < centroid[0] + 1) & (bigBox[:, 1] < centroid[1] + 1) & (bigBox[:,2] > centroid[2])
#    regionMasks[4] = (bigBox[:, 0] > centroid[0]) & (bigBox[:, 1] > centroid[1]) & (bigBox[:,2] < centroid[2] + 1)
#    regionMasks[5] = (bigBox[:, 0] > centroid[0]) & (bigBox[:, 1] < centroid[1] + 1) & (bigBox[:,2] > centroid[2])
#    regionMasks[6] = (bigBox[:, 0] < centroid[0] + 1) & (bigBox[:, 1] > centroid[1]) & (bigBox[:,2] > centroid[2])
#    regionMasks[7] = (bigBox[:, 0] > centroid[0]) & (bigBox[:, 1] > centroid[1]) & (bigBox[:,2] > centroid[2])
#
#    for i in range(8):
#        quadrantPoints[i] = bigBox[regionMasks[i]]
#        
##    for i in range(len(quadrantPoints[0]), np.shape(quadrantPoints[0])[0], -1):
#        
#
#    print quadrantPoints[0]
#    return quadrantPoints

def lesionQuadrants(lesion, downSample=1):
    quadrants = [[],[],[],[],[],[],[],[]]
    
    les = np.asarray(lesion)
    xMin = np.amin(les[:,0])
    xMax = np.amax(les[:,0])

    yMin = np.amin(les[:,1])
    yMax = np.amax(les[:,1])
    
    zMin = np.amin(les[:,2])
    zMax = np.amax(les[:,2])
    
    xc, yc, zc = [int(np.mean(xx)) for xx in zip(*lesion)]
    
    for i in range(xc, xMax, downSample):
        for j in range(yc, yMax, downSample):
            for k in range(zc, zMax, 1):
                if [i,j,k] in lesion:
                    quadrants[0].append([i,j,k])
                
            for k in range(zc, zMin, -1):
                if [i,j,k] in lesion:
                    quadrants[1].append([i,j,k])
    
        for j in range(yc, yMin, -downSample):
            for k in range(zc, zMax, 1):
                if [i,j,k] in lesion:
                    quadrants[2].append([i, j, k])
            for k in range(zc, zMin, -1):
                if [i,j,k] in lesion:
                    quadrants[3].append([i,j,k])
        
    for i in range(xc, xMin, -downSample):
        for j in range(yc, yMax, downSample):
            for k in range(zc, zMax, 1):
                if [i,j,k] in lesion:
                    quadrants[4].append([i,j,k])
                
            for k in range(zc, zMin, -1):
                if [i,j,k] in lesion:
                    quadrants[5].append([i,j,k])
    
        for j in range(yc, yMin, -downSample):
            for k in range(zc, zMax, 1):
                if [i,j,k] in lesion:
                    quadrants[6].append([i, j, k])
            for k in range(zc, zMin, -1):
                if [i,j,k] in lesion:
                    quadrants[7].append([i,j,k])
    
    for q in quadrants:
        if len(q) == 0:
            q.append([xc,yc,zc])
    return quadrants


def getLBPFeatures(scan, boundingBoxes, images):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
        
    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)
        
        #centroid
#        centroidX, centroidY, centroidZ = [int(np.mean(x)) for x in zip(*lesionPoints)]
#
#        
#        if (len(lesionPoints) > 2) & (len(lesionPoints) < 11):
#            box = boundingBoxes['tiny']
#            print 'using tiny bounding box'
#        if (len(lesionPoints) > 10) & (len(lesionPoints) < 26):
#            box = boundingBoxes['small']
#            print 'using small bounding box'
#        if (len(lesionPoints) > 25) & (len(lesionPoints) < 101):
#            box = boundingBoxes['medium']
#            print 'using med bounding box'
#        if (len(lesionPoints) > 100):
#            box = boundingBoxes['large']
#            print 'using lg bounding box'
#        if (len(lesionPoints) < 3):
#            continue
#        
#        regions = getLBPRegions(box, [centroidX, centroidY, centroidZ])
        
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
#                lbpHist = np.zeros((len(regions), 2**(lbpBinsTheta*lbpBinsPhi)), np.uint8)
#                for i, reg in enumerate(regions):
#                    lbpHist[i, :] = rotationInvariantLBP(images[mod], reg, radius)
#                    
#                lbpVector = np.hstack((lbpHist[0], lbpHist[1], lbpHist[2], lbpHist[3], lbpHist[4], lbpHist[5], lbpHist[6], lbpHist[7]))
#                del lbpHist
            
            
#                lbpHist = simpleLBP(images[mod], lesion, radius)
                feature[r, ...] = uniformLBP(images[mod], lesion, radius)
                
#                lbpVector = lbpTop(images[mod], lesionPoints, radius)
                saveDocument[mod] = Binary(pickle.dumps(feature, protocol=2))
     

        for i in range(30):
            try:
#                db['lbp'].insert_one({'_id': scan.uid+'_'+str(l)})
                db['lbp'].update_one({'_id' : scan.uid + '_' + str(l)}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect as error:
                print error
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['MSLAQ']
                time.sleep(2*i)

def getGaborFeatures(scan, gabors, nBest, images):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    imageStartTime = time.time()
    
    modalities = ['t1p', 't2w', 'pdw', 'flr']
    rotTheta = np.linspace(0,180, num=4, endpoint=False)
    rotPhi = np.linspace(0,90, num=2, endpoint=False)

    gaborWidth = [1, 2.5, 4.5]
    gaborSpacing = [0.01, 0.2, 2]
    
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
            
def getIntensityFeatures(scan, images):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    
    intensityMin = {"t1p": 32.0, "t2w": 10.0, "flr": 33.0, "pdw": 49.0}
    intensityMax = {'t1p': 1025.0, 't2w': 1000.0, 'flr': 1016.0, 'pdw': 1018.0}

    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)
        
        histBins = 4
        
        for m in modalities:
            intensities = []
            for point in lesion:
                intensities.append(images[m][point[0], point[1], point[2]])
            
            intensityHist = np.histogram(intensities, histBins, (intensityMin[m], intensityMax[m]))
            intensityHist = intensityHist[0] / np.sum(intensityHist[0], dtype='float')
            
            if np.isnan(intensityHist).any():
                intensityHist = np.zeros((histBins))
                intensityHist[0] = 1
            
            filteredIntensityHist = gaussian_filter(intensityHist, float(histBins)/30.0)
#            print 'intensity', m, filteredIntensityHist
#            print 'intensity', m, intensityHist
            saveDocument[m] = Binary(pickle.dumps([np.mean(intensities), np.var(intensities)], protocol=2))
        
        for i in range(30):
            try:
                db['intensity'].update_one({'_id' : scan.uid + '_' + str(l)}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect:
                print 'error writing results, tyring again'
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['intensity']
                time.sleep(2*i)	
        			

def getFeaturesOfList(mri_list, gabors, riftRegions, boundingBoxes):
    for i, scan in enumerate(mri_list):
        images = {}
        for j, m in enumerate(modalities):
            images[m] = nib.load(scan.images[m]).get_data()
        
        print scan.uid, i, '/', len(mri_list)
        startTime = time.time()
        
        sys.stdout.flush()
        if doContext:
            getICBMContext(scan, images)

        if doGabors:
            getGaborFeatures(scan, gabors, 10, images)

        if doLBP:
            getLBPFeatures(scan, boundingBoxes, images)
        
        if doRIFT:
            getRIFTFeatures2D(scan, riftRegions, images)
            
        if doShape:
            getShapeFeatures(scan)
        
        if doIntensity:
            getIntensityFeatures(scan, images)
            
        elapsed = time.time() - startTime
        print elapsed, "seconds"

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
    
#    mri_list = convertToNifti(mri_list)
#    mri_list = gzipNiftiFiles(mri_list)
    
#    mri_list = invertLesionCoordinates(mri_list)
    boundingBoxes = getBoundingBox(mri_list)   
    gabors = generateGabors()    
    riftRegions = generateRIFTRegions2D(riftRadii)
    
    
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
    
    lesionsInBrain = []
    
    lesionBins = [0, 0, 0, 0, 0]

    fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,6))    
    
    for scan in mri_list:
        lesionsInBrain.append(len(scan.lesionList))
        
        
        for lesion in scan.lesionList:
            if len(lesion) < 3:
                lesionBins[0] += 1
            elif len(lesion) <= 10:
                lesionBins[1] += 1
            elif len(lesion) <= 25:
                lesionBins[2] += 1
            elif len(lesion) <= 100:
                lesionBins[3] += 1
            else:
                lesionBins[4] += 1

            
    ax.bar(range(len(lesionBins)), lesionBins, 0.35)
    ax.set_title('Distribution of Lesion Sizes')
    xticks = ['< 3', '3-10', '11-25', '25-100', '> 101']
    ax.set_xlabel('Lesion Size (voxels)')
    ax.set_xticks(np.add(range(len(xticks)), 0.2))
    ax.set_xticklabels(xticks)
    
    ax2.hist(lesionsInBrain)
    ax2.set_title('Distribution of Lesions per Brain')
    ax2.set_xlabel('Lesions in Brain')
    
    plt.tight_layout()
    plt.show()
    
    for t in scan.tissues:    
#    for mod in modalities:
        intensities = []
        context = []
        for scan in mri_list:

            mri_data = nib.load(scan.priors[t]).get_data()
#            
#            
            for lesion in scan.lesionList:
                for p in lesion:
                    context.append(mri_data[p[0],p[1],p[2]])
#            print t, np.min(context), np.max(context)

#            mri_data = nib.load(scan.images[mod]).get_data()
#            plt.imshow(mri_data[:,:,30].T, cmap = cm.Greys_r)
#            plt.axis('off')        
#            plt.show()
            
            
#            for lesion in scan.lesionList:
#                for p in lesion:
#                    intensities.append(mri_data[p[0], p[1], p[2]])
        
        
#        print mod, np.min(intensities), np.max(intensities)
        print t, np.min(context), np.max(context)


def getShapeFeatures(scan):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']    
    
    bins = np.linspace(0, 20, num=21)    
    
    for l, lesion in enumerate(scan.lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)        

        img = np.zeros((256, 256, 60), dtype='float')
                
        for point in lesion:
            img[point[0], point[1], point[2]] = 1
                
#        centroid = center_of_mass(img)
        boundaryDistance = distance_transform_edt(img, sampling=[1, 1, 3])

        #perimeterPoints = img[boundaryDistance==1]
        #print perimeterPoints
        #print np.shape(perimeterPoints)[0]

        #distances = []
        #for perimeterPoint in perimeterPoints:
        #    distances.append(distance.euclidean(perimeterPoint, centroid))
            
        #sphereness = np.mean(distances) / np.std(distances)

        #les = np.asarray(lesion)
        #print les
        #xRange = np.amax(les[:,0]) - np.amin(les[:,0]) + 1
        #yRange = np.amax(les[:,1]) - np.amin(les[:,1]) + 1
        #zRange = np.amax(les[:,2]) - np.amin(les[:,2]) + 1

        #rectangularity = float(len(lesion)) / float(xRange*yRange*zRange)

        #hull = ConvexHull(lesion)
        #convexity = float(hull.Area / len(perimeterPoints))
        #solidity = float(len(lesion) / hull.Volume)
           
        shapeHistogram, binEdges = np.histogram(boundaryDistance[boundaryDistance > 0], bins=bins, normed=True)    

        #saveDocument['sphereness'] = Binary(sphereness)
        #saveDocument['rectangularity'] = Binary(rectangularity)
        #saveDocument['convexity'] = Binary(convexity)
        #saveDocument['solidity'] = Binary(solidity)
        saveDocument['shapeHistogram'] = Binary(pickle.dumps(shapeHistogram))
            
        for i in range(30):
            try:
                db['shape'].update_one({'_id' : scan.uid + '_' + str(l)}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect:
                print 'error writing results, tyring again'
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['shape']
                time.sleep(2*i)
                    
if __name__ == "__main__":
#    infile = open('/usr/local/data/adoyle/mri_list.pkl', 'rb')
#    mri_list = pickle.load(infile)
#    
#    for i, scan in enumerate(mri_list[0:2]):
#        getICBMContext(scan)
#        print i, '/', len(mri_list)


#    displayGabors() 
    main()
    import analyze_lesions
    analyze_lesions.justTreatmentGroups()


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