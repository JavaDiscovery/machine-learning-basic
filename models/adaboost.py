#coding=utf-8
'''
Created on Nov 17, 2015

@author: zxf
'''
from numpy import *

def loadSimpleData():
    dataMat=mat([[1,2.1],
                 [2,1.1],
                 [1.3,1],
                 [1,1],
                 [2,1]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    input:
        threshIneq: lt:less than or gt:great than
    """
    retArray=ones((dataMatrix.shape[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else :
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    """
    input:
        D：m*1：每一个样本的权值, 
    output:
        bestStump: {'dim':i,'thresh':阈值，‘ineq':'lt'or'gt'}
        minErr: 最小的误差
        bestClassEst: 最好的样本分类标签列表
    """
    dataMatrix=mat(dataArr);labelMat=mat(classLabels).T
    m,n=dataMatrix.shape
    numStep=10.0;bestStump={};bestClassEst=mat(zeros((m,1)))
    minErr=inf
    for i in range(n):
        rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numStep    #10个阈值，每个阈值之间的距离
        for j in range(1,int(numStep)+1):
            for inequal in ['lt','gt']:
                threshVal=rangeMin+float(j)*stepSize
                predictedVals=stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0
                #得到了以第I个特征，阈值为threshVal=rangeMin+float(j)*stepSize时，的全部样本的误差
                weightedError=D.T*errArr
                if weightedError<minErr:
                    minErr=weightedError
                    bestClassEst=predictedVals
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minErr,bestClassEst
 
def adaBoostTrain(dataArr,classLabels,numIter=40):
    weakClassifyArr=[]
    m=shape(dataArr)[0] #这个时候传进来的数据是list形式，并不是matrix或者array
    D=mat(ones((m,1))/m) #初始的样本权重矩阵，表示训练数据中每个样本的权重
    aggClassEst=mat(zeros((m,1)))
    for i in range(numIter):
        bestStump,error,classEst=buildStump(dataArr, classLabels, D)
        print "D:",D.T
        alpha=float(0.5*log((1-error)/max(error,1e-16)))#不能让分母为0！！！
        bestStump['alpha']=alpha
        #完成了一次初始弱分类器的添加
        weakClassifyArr.append(bestStump)
        print "classEst:",classEst.T
        temp=multiply((-alpha*mat(classLabels).T),classEst)
        D=multiply(D,exp(temp))
        D=D/D.sum()
        #分类器的权重与分类器的分类结果的乘积的累加
        aggClassEst+=alpha*classEst
        lastClassEst=sign(aggClassEst)
        #分类错误的样本数目
        aggErrors=sum(lastClassEst!=mat(classLabels).T)
#         aggErrors=multiply(lastClassEst!=mat(classLabels).T,ones((m,1)))
        errorRate=float(aggErrors)/m
        print "error rate:",errorRate,"\n"
        if errorRate==0:
            break
    return weakClassifyArr
        

if __name__=='__main__':
    dataMat,classLabels=loadSimpleData()
    D=mat(ones((5,1))/5)
#     print buildStump(dataMat, classLabels, D)
    print adaBoostTrain(dataMat, classLabels, 9)
