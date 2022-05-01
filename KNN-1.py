import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

def readfile(filename):
   fr = open(filename)
   linenumber = len(fr.readlines())
   returnMat = np.zeros((linenumber, 3))
   classLabelVector = []  # prepare labels return
   fr = open(filename)
   index = 0
   for line in fr.readlines():
       line = line.strip()
       listFromLine = line.split('\t')
       returnMat[index,:] = listFromLine[0:3]
       classLabelVector.append(int(listFromLine[-1]))
       index += 1
   return returnMat, classLabelVector

def showdatas(datingDataMat, datingLabels):

    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(20,12))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    axs0_title_text = axs[0][0].set_title(u'Sheet 1')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'fly miles per year')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'video games time percent')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title(u'Sheet 2')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'fly miles per year')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'ice cream cost per week')
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    axs2_title_text = axs[1][0].set_title(u'Sheet 3')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'cideo games time percent')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'ice cream cost per week')
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    plt.show()

def autonorm(datamats):
    minv,maxv = [],[]
    for k in range(0,3):
        minval,maxval = datamats[0][k],datamats[0][k]
        for i in range(0,len(datamats)):
            if datamats[i][k] >= maxval:
                maxval = datamats[i][k]
            if datamats[i][k] <= minval:
                minval = datamats[i][k]
        minv.append(minval)
        maxv.append(maxval)
        for j in range(0,len(datamats)):
            datamats[j][k] = (datamats[j][k]-minval)/(maxval - minval)
    return datamats,minv,maxv

def classify(k,datanorm,testnorm,dataclassify):
    value = np.zeros(3*len(datanorm),dtype=np.int32)
    answer = []
    for j in range(0,3):
        for i in range(0,len(datanorm)):
            value[i] += (testnorm[j] - datanorm[i][j])**2
    min = 100
    minget = 0
    for w in range(0,k):
        for i in range(0,len(value)):
            if value[i] >= min:
                minget = i
        np.delete(value,minget,axis=0)
        answer.append(dataclassify[minget])
    return np.argmax(np.bincount(answer))


def testnorm(testdata,maxv,minv):
    testnorm = []
    for i in range(0,3):
        a = (testdata[0][i]-minv[i])/(maxv[i]-minv[i])
        testnorm.append(a)
    return testnorm

data,dataclassify = readfile('datingTestSet2.txt')
k = 10       #set k by yourself
datanorm,minv,maxv = autonorm(data)
showdatas(data,dataclassify)
tests,testclassify = readfile('testfile.txt')
testnorm = testnorm(tests,maxv,minv)
print(classify(k,datanorm,testnorm,dataclassify))