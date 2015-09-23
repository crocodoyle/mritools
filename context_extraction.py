import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

from mri import mri
#from scipy import stats

#from scipy.ndimage.filters import convolve
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.interpolation import map_coordinates

import cPickle as pickle

import subprocess
import simplejson as json


#import transformations as transforms
import time

import redis

from random import shuffle

#import sys


data_dir = '/usr/local/data/adoyle/trials/MS-LAQ-302-STX/'
threads = 8
recompute = False



    
def gabor(rotationTheta, rotationPhi, sinSpacing, gaussianWidth):
    sig = gaussianWidth
    sig_x = sig
    sig_y = sig
    sig_z = sig
    
    window_size = 10
#    z_window_size = 4

    gabor_range = range(-window_size/2, window_size/2+1)
#    x = range(-window_size/2,window_size/2+1)
#    y = range(-window_size/2,window_size/2+1)
#    z = range(-z_window_size/2,z_window_size/2+1)
    
    gaus = np.zeros((11,11,11))
    sine = np.zeros((11,11,11))
    
#    u = sinSpacing*np.sin(rotationTheta)*np.cos(rotationPhi)
#    v = sinSpacing*np.sin(rotationTheta)*np.sin(rotationPhi)    
#    w = sinSpacing*np.cos(rotationTheta)
#    
#    S = np.sqrt(u**2 + v**2 + w**2)
    
    for i, m in enumerate(gabor_range):
        for j, n in enumerate(gabor_range):
            for k, o in enumerate(gabor_range):
                gaus[i,j,k] = (1/((2*np.pi**1.5)*sig_x*sig_y*sig_z))*np.exp(-((m/sig_x)**2 + (n/sig_y)**2 + (o/sig_z)**2)/2)
#                sine[i,j,k] = np.real(S*np.exp(2j*np.pi*(u*m + v*n + w*o)))
                sine[i,j,k] = np.real(np.exp(2j*np.pi*sinSpacing*np.sqrt(m**2 + 1 + o**2)))
    
    gab = np.multiply(gaus, sine)
    gab = np.divide(gab, np.max(gab))
        
    gab = rotate(gab, rotationTheta, axes=(0,1), reshape=False, output=None, order=1, mode='constant', cval=0.0, prefilter=True)
    gab = rotate(gab, rotationPhi, axes=(1,2), reshape=False, output=None, order=1, mode='constant', cval=0.0, prefilter=True)    
    gab = zoom(gab, [1, 1, 60.0/256.0])
    
    return gab  

def lbp(image, lesionPoints, radius):
    lbpPatterns = np.zeros((len(lesionPoints), 8*4), dtype='float')
    
    for l, [x, y, z] in enumerate(lesionPoints):
        sampleAt = lbpSphere(radius, [x,y,z], 8, 4)        
        sampledPoints = map_coordinates(image, sampleAt, order=1)
        
        for m, sampledValue in enumerate(sampledPoints):
            if image[x,y,z] > sampledValue:
                lbpPatterns[l, m] = 1
        
    lbpFeature = np.zeros((32), dtype='float')
    
    for l in range(np.shape(lesionPoints)[0]):
        lbpFeature += lbpPatterns[l,:] / float(len(lesionPoints))

    return lbpFeature

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
                    print total
                    
#                    print scan.images['t1p']
    return mri_list


def lbpSphere(radius, centre, pointsTheta, pointsPhi):
    sampleAt = []
    theta = np.linspace(0, np.pi, num=pointsTheta, endpoint=False)
    phi = np.linspace(0, np.pi*2, num=pointsPhi, endpoint=False)
    
    for t in theta:
        for p in phi:
            x = radius*np.sin(p)*np.cos(t) + centre[0]
            y = radius*np.sin(p)*np.sin(t) + centre[1]
            z = radius*np.cos(p)*(60.0/256.0) + centre[2]
            
            sampleAt.append((x,y,z))
    
    return np.asarray(np.transpose(sampleAt))


def getLBPFeatures(scan):
    red = redis.StrictRedis(host='localhost', port=6379, db=0)

    imageStartTime = time.time()

    if len(scan.lesionList) == 0:
        lesionList = scan.separateLesions()
    else:
        lesionList = scan.lesionList
    
    images = {}
    
    modalities = ['t1p', 't2w', 'pdw', 'flr']
        
    lbpRadii = [1, 2, 3]
    
    for j, m in enumerate(modalities):
        print scan.uid, m
        images[m] = nib.load(scan.images[m]).get_data()

        for r, radius in enumerate(lbpRadii):
            for l, lesionPoints in enumerate(lesionList):
                lbpVector = lbp(images[m], lesionPoints, radius)
                red.set(scan.uid + ':lbp:' + m + ':scale' + str(radius) + ':lesion' + str(l), json.dumps(lbpVector.tolist()))


    imageEndTime = time.time()
    elapsed = imageEndTime - imageStartTime
    print elapsed/(60), "minutes", elapsed%60, "seconds"

def getGaborFeatures(scan):
    red = redis.StrictRedis(host='localhost', port=6379, db=0)

    imageStartTime = time.time()

    if len(scan.lesionList) == 0:
        lesionList = scan.separateLesions()
    else:
        lesionList = scan.lesionList
    
    modalities = ['t1p', 't2w', 'pdw', 'flr']
    rotTheta = np.linspace(0,180, num=4, endpoint=False)
    rotPhi = np.linspace(0,90, num=2, endpoint=False)


    gaborWidth = [1, 2.5, 4.5]
    gaborSpacing = [0.01, 0.2, 2]
    
    images = {}
    
    for j, m in enumerate(modalities):
        print scan.uid, m
        if not recompute and red.exists(scan.uid + ':gabResponseScore:' + str(m) + ':lesion:0'):
            break
        images[m] = nib.load(scan.images[m]).get_data()
        
        for l, lesionPoints in enumerate(lesionList):

            
            gaborResponses = np.zeros((len(lesionPoints), len(rotTheta), len(rotPhi), len(gaborWidth), len(gaborSpacing)))
            for i, [x, y, z] in enumerate(lesionPoints):
                for r, rot in enumerate(rotTheta):
                    for r2, rot2 in enumerate(rotPhi):
                        for n, width in enumerate(gaborWidth):
                            for o, spacing in enumerate(gaborSpacing):
                                gab = gabor(rot, rot2, spacing, width)
                                gaborResponses[i, r, r2, n, o] = np.sum(np.multiply(images[m][x-5:x+6,y-5:y+6,z-1:z+2], gab))
            
            
            toWrite = json.dumps(np.ndarray.flatten(gaborResponses).tolist())
            key = scan.uid + ':gabResponseScore:' + str(m) + ':lesion:' + str(l)
            red.set(key, toWrite)


    imageEndTime = time.time()
    elapsed = imageEndTime - imageStartTime
    print elapsed/(60), "minutes"


def getFeaturesOfList(mri_list):
    for scan in mri_list:
        getGaborFeatures(scan)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    shuffle(l)
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
    

def main():
    reload_list = False
    multithreaded = True
    
    #malf_classes = ['bg', 'bs', 'cgm', 'crblr_gm', 'crblr_wm', 'csf', 'dgm', 'lv', 'ov', 'wm']
    #good_malf_classes = ['cgm', 'dgm', 'wm']
    #malf_tissues = {}
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
    
    
    if multithreaded:
        chunkSize = (len(mri_list) / threads) + 1
        
        procs = []        
        for i, sublist in enumerate(chunks(mri_list, chunkSize)):
            print 'Starting process', i, 'with', len(sublist), 'images to process'
            worker = mp.Process(target=getFeaturesOfList, args=(sublist,))
            worker.start()
            procs.append(worker)
        
        print 'started processes'    
        
        for proc in procs:
            print 'Waiting...'
            proc.join()
    else:  
        for i, scan in enumerate(mri_list[0:2]):
            getGaborFeatures(scan)

    print 'Done'
    
    endTime = time.time()
    
    elapsed = endTime - startTime
    print "Total time elapsed:", elapsed/3600, 'hours', elapsed/60, 'minutes'
        
    
if __name__ == "__main__":
    main()
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
