#coding=utf-8
'''
Created on Nov 16, 2015

@author: zxf
'''
#加载数据，默认只有两个特征值
from numpy import *
from matplotlib.pyplot import xlabel, ylabel, show, figure
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split();
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).T #N行1列
    m,n=shape(dataMatrix)
    alpha=0.001
    maxIterat=500
    weights=ones((n,1))#特征权重的初始值
    for k in range(maxIterat):
        h=sigmoid(dataMatrix*weights)#矩阵相乘
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.T*error
    return weights

def plotBestFit(weights):
#     import matplotlib.pyplot as plt
    weights=weights.getA()#得到全部值，不会改变原来的值
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        #加载数据：类别为1的放在xcord1中，类别为0的放在xcord2中
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]#1+w1*x+w2*y=0
    ax.plot(x,y)
    xlabel('X1');ylabel('X2')
    show()

def stocGradAscent(dataMatrix,classLabels,numIter=150):#classLabels:1行n列
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    #增加迭代
    for j in range(numIter):
        dataIndex=range(m)#随机选择某一样本数据更新
        for i in range(m):
            #每次迭代，一个样本就是一次调整w，一次调整w就改变alpha！为什么用这个公式？？？？（避免参数的严格下降）
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]#随机梯度算法中的偏导数？？？？？
            del(dataIndex[randIndex])#不重复的随机选择
    return weights

if __name__=="__main__":
    dataArr,labelArr=loadDataSet()
    weights=gradAscent(dataArr, labelArr)
    weights2=stocGradAscent(array(dataArr), labelArr)
    print weights
    print weights2
    plotBestFit(mat(weights2).T)
    
