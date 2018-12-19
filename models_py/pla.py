#coding=utf-8
'''
Created on Nov 19, 2015

@author: zxf
'''

from numpy import *

def loadDataSet(filename):
    dataArr=[];labelArr=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataArr.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelArr.append(int(lineArr[2]))
    return dataArr,labelArr
#判断两个numpy数组是否全部为TRUE
def trainPLA(dataSet,labels):
    dataMat=mat(dataSet);labelsMat=mat(labels)
    m,n=dataMat.shape
    w=[0]*n;error=0
    #python 模拟do-while 语句
    while True:
        error=0;minError=1000000
        for i in xrange(m):
            data=dataMat[i];label=labels[i]
            score=data*(mat(w).T)
            sign=1 if score>0 else -1
            if sign!=label:
                error+=1
                w+=label*data
        for wt in w:
            print wt#输出w的变化
        print '\nerror:'+str(error)
        print '------'
        if error==0:
            break
    return w
        

def calcErrorCount(dataSet,labels,wTemp):
    dataMat=mat(dataSet);labelsMat=mat(labels)
    m,n=dataMat.shape
    error=0
    for i in range(m):
        data=dataMat[i];label=labels[i]
        score=data*(mat(wTemp).T)
        sign=1 if score>0 else -1
        if sign!=int(label):
            error+=1
    return error


def trainPocketPLA(dataSet,labels):
    dataMat=mat(dataSet);labelsMat=mat(labels)
    m,n=dataMat.shape
    w=[0]*n;minErrorCount=1000000
    for i in range(100):#运行50次还有错。
        randomIndex=random.uniform(0,m)
        data=dataMat[int(randomIndex)];label=labels[int(randomIndex)]
        score=data*(mat(w).T)
        sign=1 if score>0 else -1
        if sign!=label:
            wTemp=w;
            wTemp+=label*data
            if calcErrorCount(dataSet,labels,wTemp)<minErrorCount:
                w=wTemp
        for wt in w:
            print wt
    return w
        
if __name__ == '__main__':
    dataSet,labels=loadDataSet('train.dat')
    trainPocketPLA(dataSet, labels)