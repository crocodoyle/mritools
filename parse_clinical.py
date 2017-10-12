# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:33:35 2016

@author: adoyle
"""

import csv 
import time
import cPickle as pkl

from pymongo import MongoClient



dbIP = '132.206.73.115'
dbPort = 27017


dbClient = MongoClient(dbIP, dbPort)
db = dbClient['MSLAQ']


csvfile = open('/usr/local/data/adoyle/MSLAQ-clinical.csv', 'rb')
mri_list = pkl.load(open('/usr/local/data/adoyle/mri_list.pkl', 'rb'))

csvwriter = csv.writer(open('/usr/local/data/adoyle/extraOnes.csv', 'wb'))

csvreader = csv.reader(csvfile)

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
                print uid, 'found!'
        
        if not inList:
            print uid, 'not found!'
            csvwriter.writerow([uid[0:3] + '_' + uid[4:]])
        
        print uid, treatment, newT2, newT1, atrophy        
        
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

        for i in range(30):
            try:
                db['clinical'].update_one({'_id' : uid}, {"$set": saveDocument}, upsert=True)
                break
            except pymongo.errors.AutoReconnect:
                dbClient = MongoClient(dbIP, dbPort)
                db = dbClient['MSLAQ']
                time.sleep(2*i)
    index +=1