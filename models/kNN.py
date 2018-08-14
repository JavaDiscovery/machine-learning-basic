#coding=utf-8
'''
Created on Nov 4, 2015

@author: zxf
'''
from numpy import *
import operator
import os

def createDataSet():
    group=array([[1,0.9],[1,1],[0.1,0.2],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def kNNClassify(newInput,dataSet,labels,k):
    numSample=dataSet.shape[0]
    #第一步：计算欧式距离
    diff=tile(newInput, (numSample,1))-dataSet#tile 将newinput变成numSample行1列
    squaredDiff=diff**2
    squaredDist=sum(squaredDiff,axis=1)#axis=0:操作列；axis=1:操作行
    distance=squaredDist**0.5
    #第二步：保存距离
    sortedDistIndices=argsort(distance)
    classCount={}
    for i in xrange(k):
         #第三步：选择最小的k距离
        voteLabel=labels[sortedDistIndices[i]]
        #第四步：计算最小的k距离样本中每个类别出现的次数
        classCount[voteLabel]=classCount.get(voteLabel,0)+1#!!!!get(key[,default])
    #第五步：返回出现次数最多的那个类别即为newInput的类别
    maxCount=0
    for key,value in classCount.items():
        if value>maxCount:
            maxCount=value 
            maxIndex=key #dictionary中的key就是数据的类别
    return maxIndex
        
#将数字图像转换成向量
def img2vector(filename):
    rows,cols=32,32
    imgVector=zeros((1,rows*cols))
    fileIn=open(filename)
    for row in xrange(rows):
        lineStr=fileIn.readline()
        for col in xrange(cols):
            imgVector[0,row*32+col]=int(lineStr[col])
    return imgVector

#加载数据，调用img2vector
def loadDataSet():
    #训练数据
    dataSetDir='/home/zxf/workspace/pylearn/ml/knn/'
    trainingFileList=os.listdir(dataSetDir+'trainingDigits')
    numSample=len(trainingFileList)#样本的数量，每个数字200个样本，共计20000个
    train_x=zeros((numSample,1024))#一个样本是一个1*1024的向量
    train_y=[]
    for i in xrange(numSample):
        filename=trainingFileList[i]
        train_x[i]=img2vector(dataSetDir+'trainingDigits/%s' % filename)
        label=int(filename.split('_')[0])
        train_y.append(label)
    #测试数据
    testFileList=os.listdir(dataSetDir+'testDigits')
    numSample=len(testFileList)
    test_x=zeros((numSample,1024))
    test_y=[]
    for i in xrange(numSample):
        filename=testFileList[i]
        test_x[i]=img2vector(dataSetDir+'testDigits/%s'%filename)
        label=int(filename.split('_')[0])
        test_y.append(label)
    return train_x,train_y,test_x,test_y

#测试代码
def testHandWriting ():
    train_x,train_y,test_x,test_y=loadDataSet()
    numTestSamples=test_x.shape[0]
    matchCount=0
    for i in xrange(numTestSamples):
        predict=kNNClassify(test_x[i], train_x, train_y, 3)
        if predict==test_y[i]:
            matchCount+=1
    accuracy=float(matchCount)/numTestSamples
    print accuracy*100
            
if __name__=='__main__':
    testHandWriting()
    
    
        
        
        
        
        
        
               
        
        
        
        
        
        
        
        