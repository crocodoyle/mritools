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

import cPickle as pickle

import subprocess
import time, sys

from random import shuffle
from pymongo import MongoClient
from bson.binary import Binary
import pymongo

icbmRoot = '/usr/local/data/adoyle/trials/quarantine/common/models/icbm_avg_152_'
data_dir = '/usr/local/data/adoyle/trials/MS-LAQ-302-STX/'
lesion_atlas = '/usr/local/data/adoyle/trials/quarantine/common/models/icbm_avg_3714_t2les.mnc.gz'


threads = 8
recompute = False
reconstruct = True

doGabors = False
doLBP = False
doContext = True
indexLesionPosition = True
doRIFT = False

reload_list = False
multithreaded = False


modalities = ['t1p', 't2w', 'pdw', 'flr']


#blizzard ip
dbIP = '132.206.73.115'
dbPort = 27017

def generateGabors():
    rotTheta = np.linspace(0,180, num=4, endpoint=False)
    rotPhi = np.linspace(0,90, num=2, endpoint=False)

    gaborWidth = [1, 2.5, 4.5]
    gaborSpacing = [0.1, 0.2, 1]    
    
    gabors = np.zeros((len(rotTheta), len(rotPhi), len(gaborSpacing), len(gaborWidth), 11, 11, 3))
    
    for r, rot in enumerate(rotTheta):
        for r2, rot2 in enumerate(rotPhi):
            for n, width in enumerate(gaborWidth):
                for o, spacing in enumerate(gaborSpacing):
                    gabors[r, r2, n, o, :, :, :] = gabor(rot, rot2, spacing, width)
                    
    return gabors


def gabor(rotationTheta, rotationPhi, sinSpacing, gaussianWidth):
    sig = gaussianWidth
    sig_x = sig
    sig_y = sig
    sig_z = sig
    
    window_size = 10

    gabor_range = range(-window_size/2, window_size/2+1)
    
    gaus = np.zeros((11,11,11))
    sine = np.zeros((11,11,11))
        
    for i, m in enumerate(gabor_range):
        for j, n in enumerate(gabor_range):
            for k, o in enumerate(gabor_range):
                gaus[i,j,k] = (1/((2*np.pi**1.5)*sig_x*sig_y*sig_z))*np.exp(-((m/sig_x)**2 + (n/sig_y)**2 + (o/sig_z)**2)/2)
#                sine[i,j,k] = np.real(S*np.exp(2j*np.pi*(u*m + v*n + w*o)))
                sine[i,j,k] = np.real(np.exp(2j*np.pi*sinSpacing*np.sqrt(m**2 + 1 + 1)))
    
    gab = np.multiply(gaus, sine)
    gab = np.divide(gab, np.max(gab))
        
    gab = rotate(gab, rotationTheta, axes=(0,1), reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
    gab = rotate(gab, rotationPhi, axes=(1,2), reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=True)    
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

#generates a lookup table of points to sample for RIFT function
def generateRIFTRegions(radii):
    pointLists = []
    
    for r in range(len(radii)):
        pointLists.append([])
        
    for x in range(-np.max(radii), np.max(radii)):
        for y in range(-np.max(radii), np.max(radii)):
            for z in range(-np.max(radii), np.max(radii)):
                distance = np.sqrt(x**2 + y**2 + (z*60.0/256.0)**2)
                
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
    
    for t in theta:
        for p in phi:
            x = radius*np.sin(p)*np.cos(t) + centre[0]
            y = radius*np.sin(p)*np.sin(t) + centre[1]
            z = radius*np.cos(p)*(60.0/256.0) + centre[2]
            
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
        
            
            
def getLBPFeatures(scan):
    dbClient = MongoClient(dbIP, dbPort)
    db = dbClient['MSLAQ']
    
    imageStartTime = time.time()

    if len(scan.lesionList) == 0:
        lesionList = scan.separateLesions()
    else:
        lesionList = scan.lesionList
    
    images = {}

    lbpRadii = [1, 2, 3]
    
    for j, mod in enumerate(modalities):
        print scan.uid, mod
        images[mod] = nib.load(scan.images[mod]).get_data()

    for l, lesionPoints in enumerate(lesionList):
        saveDocument = {}
        saveDocument['_id'] = scan.uid + '_' + str(l)
            
        
        for j, mod in enumerate(modalities):
            saveDocument[mod] = {}
            
            for r, radius in enumerate(lbpRadii):
                lbpVector = lbp(images[mod], lesionPoints, radius)
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


def getFeaturesOfList(mri_list, gabors):
    riftRegions = generateRIFTRegions([2,4,6])
    
    for i, scan in enumerate(mri_list):
        print scan.uid, i, '/', len(mri_list)
        if doContext:
            getICBMContext(scan)

        if doGabors:
            getGaborFeatures(scan, gabors, 10)

        if doLBP:
            getLBPFeatures(scan)
        
        if doRIFT:
            getRIFTFeatures(scan, riftRegions)


def chunks(l, n):
    shuffle(l)
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
    

def main():
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
        
#        if reconstruct:        
#            new_list = []        
#            for scan in mri_list:
#                
#                tokens = scan.images['t1p'].split('/')[-1].split('_')                
#                
#                new_scan = scan
#                new_scan.tissues = ['csf', 'wm', 'gm', 'lesion']
#                new_scan.priors['lesion'] = new_scan.folder[0:-3] + 'stx152lsq6/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_patient_avg_ANAT-lesion-cerebrum_ISPC-stx152lsq6.mnc.gz'
#                new_scan.lesionPriorXfm = new_scan.folder + 'xfm/MS-LAQ-302-STX_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + '_m0_t1p-to-stx152lsq6.xfm'
#
#                
#
#
#                new_list.append(new_scan)
#
#            
#            mri_list = new_list
#            
#            outfile = open('/usr/local/data/adoyle/mri_list_new.pkl', 'wb')
#            pickle.dump(mri_list, outfile)
#            outfile.close()            
            
        
        infile.close()
    
    print 'MRI list loaded'
    
    
    gabors = generateGabors()    
    
    if multithreaded:
        chunkSize = (len(mri_list) / threads) + 1
        
        procs = []        
        for i, sublist in enumerate(chunks(mri_list, chunkSize)):
            print 'Starting process', i, 'with', len(sublist), 'images to process'
            worker = mp.Process(target=getFeaturesOfList, args=(sublist, gabors,))
            worker.start()
            procs.append(worker)
        
        print 'started processes'    
        
        for proc in procs:
            print 'Waiting...'
            proc.join()
    else:  
        getFeaturesOfList(mri_list, gabors)

    print 'Done'
    
    endTime = time.time()
    
    elapsed = endTime - startTime
    print "Total time elapsed:", elapsed/3600, 'hours', elapsed/60, 'minutes'

def displayGabors():
    gabors = generateGabors()
    
    for i0 in range(0,np.shape(gabors)[0]):
        for i1 in range(0,np.shape(gabors)[1]):
            for i2 in range(0,np.shape(gabors)[2]):
                for i3 in range(0,np.shape(gabors)[3]):
                    gab = gabors[i0,i1,i2,i3,:,:,1]
                    plt.subplot(9, 8, i0*3*3*2+i1*3*3+i2*3+i3 + 1)
                    plt.axis('off')
                    plt.imshow(gab)
                    plt.subplots_adjust(wspace=0.01,hspace=0.01)
                    
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
        


if __name__ == "__main__":
#    infile = open('/usr/local/data/adoyle/mri_list.pkl', 'rb')
#    mri_list = pickle.load(infile)
#    
#    for i, scan in enumerate(mri_list[0:2]):
#        getICBMContext(scan)
#        print i, '/', len(mri_list)


#    displayGabors() 
#    main()

    displayBrains()
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