#coding=utf-8
'''
Created on Nov 18, 2015

@author: zxf
'''

from numpy import *
import operator

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flipers']
    return dataSet,labels

#计算一个数据集的香农熵：以每个样本的分类结果计算：熵越大，分类结果越复杂
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featureVec in dataSet:
        currentLabel=featureVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log2(prob)
    return shannonEnt

#按照给定特征划分数据集:取出特征下标为axis的值==value的数据样本
def splitData(dataSet,axis,value):
    retDataSet=[]
    for featureVec in dataSet:
        if featureVec[axis]==value:
            reducedFeatVec=featureVec[:axis]
            reducedFeatVec.extend(featureVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        #每一个特征值的不重复的取值类型
        uniqueVals=set(featList)
        newEntropy=0.0#对每一个特征计算一次熵
        for value in uniqueVals:
            #对特征的每一种取值的分组数据，分别计算熵，再计算所有类别的期望熵，就得到新的熵的大小
            subDataSet=splitData(dataSet, i, value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    #这个函数返回的是一个元祖的列表。每一个元祖对应于一个排序的数据
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    #返回第一个元祖的第一个数据，就是类标签
    return sortedClassCount[0][0]

#用递归方法创建决策树，返回的决策树的数据结构是{bestFeatureLabel:{class1：’XX‘，class2：’XX‘...}}
def createTree(dataSet,labels):
    #第一步：排除特殊情况1：所有类别一样
    classList=[example[-1] for example in dataSet] #所有类别的list
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #第二步：排除特殊情况2：只有一个特征，根据splitDataSet()的计算过程，程序执行完了所有的特征值
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    #第三步：找到最好的特征，得到决策树，并将数据按照特征的不同值分组，再递归得到所有分组的决策树
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}#待返回的递归决策树
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueValues=set(featValues)
    for value in uniqueValues:
        subLabels=labels[:]
        #value:{key，value},key:bestFeat的一种值,value:所有同组的数据
        subDataSet=splitData(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value]=createTree(subDataSet, subLabels)
    return myTree

if __name__=='__main__':
    dataSet,labels=createDataSet()
    shannonEnt=calcShannonEnt(dataSet)
    print shannonEnt
    print dataSet
    retDataSet=splitData(dataSet, 0, 1)
    print retDataSet
    print chooseBestFeatureToSplit(dataSet)
    print createTree(dataSet, labels)
