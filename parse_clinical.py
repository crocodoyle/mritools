# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:33:35 2016

@author: adoyle
"""

import csv 
import pickle

data_dir = '/data1/users/adoyle/MS-LAQ/MS-LAQ-302-STX/'

mri_list = pickle.load(open(data_dir + 'mri_list.pkl', 'rb'))
csvwriter = csv.writer(open(data_dir + 'extraOnes.csv', 'w'))
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

        if not inList: # we don't have imaging data for the results, log it
            print(uid, 'NOT FOUND')
            csvwriter.writerow([uid[0:3] + '_' + uid[4:]])
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

    index +=1