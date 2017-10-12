# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 19:25:29 2017

@author: adoyle
"""

import matplotlib.pyplot as plt


"""
Responders DRUG A
13.0 20.0 201.0 46.0
sensitivity:  0.220338983051
specificity:  0.909502262443
Responders (certain GT)
1.0 32.0 223.0 24.0
sensitivity:  0.04
specificity:  0.874509803922
Responders (certain prediction)
23.0 46.0 175.0 36.0
sensitivity:  0.389830508475
specificity:  0.79185520362
Responders (all certain)
23.0 46.0 209.0 2.0
sensitivity:  0.92
specificity:  0.819607843137
"""

"""
Responders DRUG B
10.0 13.0 206.0 30.0
sensitivity:  0.25
specificity:  0.940639269406
Responders (certain GT)
4.0 19.0 222.0 14.0
sensitivity:  0.222222222222
specificity:  0.921161825726
Responders (certain prediction)
17.0 38.0 181.0 23.0
sensitivity:  0.425
specificity:  0.826484018265
Responders (all certain)
17.0 38.0 203.0 1.0
sensitivity:  0.944444444444
specificity:  0.842323651452
"""

#sensA = [0.220338983051, 0.04, 0.389830508475, 0.92]
#specA = [0.909502262443, 0.874509803922, 0.79185520362, 0.819607843137]
#
#sensB = [0.25, 0.222222222222, 0.425, 0.944444444444]
#specB = [0.940639269406, 0.921161825726, 0.826484018265, 0.842323651452]
#labels = ["Drug A ($\\alpha$=$\\beta$=0.5)", "Drug A ($\\alpha$=$\\beta$=0.8)", "Drug B ($\\alpha$=$\\beta$=0.5)", "Drug B ($\\alpha$=$\\beta$=0.8)"]
#
#fig = plt.figure(figsize=(4,4))
#
#p1 = plt.scatter(specA[0],sensA[0], marker='x', color='b', s=(100,))
##p2 = plt.scatter(sensA[1], specA[1], marker='x', color='c', s=(60,))
##p3 = plt.scatter(sensA[2], specA[2], marker='x', color='g', s=(60,))
#p4 = plt.scatter(specA[3], sensA[3], marker='x', color='c', s=(100,))
#
#p2 = plt.scatter(specB[0],sensB[0], marker='x', color='orange', s=(100,))
#p3 = plt.scatter(specB[3], sensB[3], marker='x', color='r', s=(100,))
#
#plt.ylabel("Sensitivity")
#plt.xlabel("Specificity")
#plt.title("Responder Prediction")
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.legend((p1, p4, p2, p3), tuple(labels), loc=3, scatterpoints=1, fancybox=True, shadow=True)
#plt.show()


fig = plt.figure(figsize=(3,3))

labels = ["NN-Euclidean", "NN-Mahalanobis", "NN-$\chi^2$", "SVM-Linear","SVM-RBF","SVM-$\chi^2$", "Random Forest", "Naive Bayes (lesion counts)"]

sens = [0.7, .66, .97, .26, .52, .45, .7, .25, .944]
spec = [.52, .4, .07, .82, .6, .62, .58, .72, .142]

p1 = plt.scatter(spec[0], sens[0], marker='x', color='b', s=(200,))
p2 = plt.scatter(spec[1], sens[1], marker='x', color='c', s=(200,))
p3 = plt.scatter(spec[2], sens[2], marker='x', color='deepskyblue', s=(200,))
p4 = plt.scatter(spec[3], sens[3], marker='+', color='gold', s=(200,))
p5 = plt.scatter(spec[4], sens[4], marker='+', color='y', s=(200,))
p6 = plt.scatter(spec[5], sens[5], marker='+', color='goldenrod', s=(200,))
p7 = plt.scatter(spec[6], sens[6], marker='*', color='r', s=(400,))
#p9 = plt.scatter(spec[8], sens[8], marker='*', color='deeppink', s=(400,))
p8 = plt.scatter(spec[7], sens[7], marker='d', color='g', s=(200,))

plt.ylabel("Sensitivity")
plt.xlabel("Specificity")
plt.title("Activity Prediction")
plt.xlim([0,1.05])
plt.ylim([0,1.05])
plt.legend((p1, p2, p3, p4, p5, p6, p7, p8), tuple(labels), loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1, fancybox=True, shadow=True)
plt.show()


#p1 = plt.scatter(sensB[0], specB[0], marker='x', color='b')
#p2 = plt.scatter(sensB[1], specB[1], marker='x', color='c')
#p3 = plt.scatter(sensB[2], specB[2], marker='x', color='g')
#p4 = plt.scatter(sensB[3], specB[3], marker='x', color='r')
#
#plt.xlabel("Sensitivity")
#plt.ylabel("Specificity")
#plt.title("Drug B Responder Prediction")
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.legend((p1, p2, p3, p4), tuple(labels), loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1, fancybox=True, shadow=True)
#plt.show()
#
#
#


fig = plt.figure(figsize=(3,3))

labels = ["Drug A ($\\beta=0.5$)", "Drug A ($\\beta=0.8$)", "Drug B ($\\beta=0.5$)", "Drug B ($\\beta=0.8$)", "Untreated $\\rightarrow$ Drug A$(\\alpha=0.5)$", "Untreated $\\rightarrow$ Drug B$(\\alpha=0.5)$"]

sens = [0.7, .944, 0.532, 0.884, 0.648, 1.0, 0.675, 0.630]
spec = [.58, .142, 0.674, 0.5, 0.593, 0.5, 0.515, 0.505]

#p1 = plt.scatter(spec[0], sens[0], marker='*', color='r', s=(200,))
#p2 = plt.scatter(spec[1], sens[1], marker='*', color='deeppink', s=(200,))
p3 = plt.scatter(spec[2], sens[2], marker='+', color='g', s=(200,))
p4 = plt.scatter(spec[3], sens[3], marker='+', color='lime', s=(200,))
p5 = plt.scatter(spec[4], sens[4], marker='x', color='b', s=(200,))
p6 = plt.scatter(spec[5], sens[5], marker='x', color='c', s=(200,))
p7 = plt.scatter(spec[6], sens[6], marker='>', color='yellowgreen', s=(200,))
p9 = plt.scatter(spec[7], sens[7], marker='>', color='lightblue', s=(200,))


plt.ylabel("Sensitivity")
plt.xlabel("Specificity")
plt.title("Activity Prediction (Treatments)")
plt.xlim([0,1.05])
plt.ylim([0,1.05])
plt.legend((p3, p4, p5, p6, p7, p9), tuple(labels), loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1, fancybox=True, shadow=True)
plt.show()


