#coding=utf-8
'''
Created on 2015年11月1日

@author: zxf
'''
from numpy import *
import matplotlib.pyplot as plt

def eucDistan(vector1,vector2):
    return sqrt(sum(power((vector1-vector2), 2)))
def initCentroids(dataSet,k):
    numSamples,dim=dataSet.shape
    centroids=zeros((k,dim))
    for i in range(k):
        index=int(random.uniform(0,numSamples))
        centroids[i,:]=dataSet[index,:]
    return centroids
def kmeans(dataSet,k):
    numSamples=dataSet.shape[0]
    clusterAssment=mat(zeros((numSamples,2)))
    clusterChanged=True
    centroids=initCentroids(dataSet, k)
    while clusterChanged:
        clusterChanged=False
        for i in xrange(numSamples):
            minDist=100000.0
            minIndex=0
            for j in range(k):
                distance=eucDistan(centroids[j,:], dataSet[i,:])
                if distance<minDist:
                    minDist=distance
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
                clusterAssment[i,:]=minIndex,minDist**2#同时更新中心点和距离
        for i in range(k):
            #nonzero(clusterAssment[:,0]==j)返回索引组成的array,得到相应的dataset中的一行一行的数据
            pointsInCluster=dataSet[nonzero(clusterAssment[:,0]==i)[0]]
            centroids[i,:]=mean(pointsInCluster, axis=0)
    print 'cluster complete!'
    return centroids,clusterAssment
def showCluster(dataSet,k,centroids,clusterAssment):
    numSamples,dim=dataSet.shape
    if dim!=2:
        print "Sorry,big than 2D"
        return 1
    #画所有的数据
    mark=['or','ob','og','^r','+r','sr','dr','<r','pr']
    if k>len(mark):
        print "Sorry,to many k"
        return 1
    for i in xrange(numSamples):
        markIndex=int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])
    #画中心点
    mark=['Dr','Db','Dg','Dk','^b','+b','sb','db','<b','pb']
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=12)
    plt.show()
                
