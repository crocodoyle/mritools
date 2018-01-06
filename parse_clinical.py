# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:33:35 2016

@author: adoyle
"""

import csv 
import pickle

data_dir = '/data1/users/adoyle/MS-LAQ/'

csvfile = open(data_dir + 'MSLAQ-clinical.csv', 'rb')
mri_list = pickle.load(open(data_dir + 'mri_list.pkl', 'rb'))

csvwriter = csv.writer(open(data_dir + 'extraOnes.csv', 'wb'))
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
                print(uid, 'found!')
        
        if not inList:
            print(uid, 'not found!')
            csvwriter.writerow([uid[0:3] + '_' + uid[4:]])
        
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

        pickle.dump(open(scan.features_dir + 'clinical' + '.pkl', 'wb'))

    index +=1