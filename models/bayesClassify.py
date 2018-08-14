#coding=utf-8
'''
Created on Nov 16, 2015

@author: zxf
'''
from numpy import *
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

#根据上面的数据集创造一个全局Vocabulary
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet) #返回的是list

#将单词列表输入按照全局单词列表映射成单词向量
def setOfWord2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print "the word:%s is not in my Vocabulary!"% word
    return returnVec

#训练分类器，得到类别为0和1的单词向量对应的概率；样本数据中类别为1的概率
def trainNB(trainMatrix,trainCategory):
    #trainMatrix:训练数据，trainCategory:样本分类标签
    numTrainDocs=len(trainMatrix)#样本的数目
    numWords=len(trainMatrix[0])#特征的数目
    pAbusive=sum(trainCategory)/float(numTrainDocs)
#     p0Num=zeros(numWords);p0Denom=0.0
    p0Num=ones(numWords);p0Denom=2  #如果一段话中出现了一个褒义词和一个贬义词，那么这段话的类别概率都是0 
#     p1Num=zeros(numWords);p1Denom=0.0
    p1Num=ones(numWords);p1Denom=2
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
#     p1Vect=p1Num/p1Denom
#     p0Vect=p0Num/p0Denom
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#将单词向量分类
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #vec2Classify:待分类的单词向量;p0Vec:类别为0的单词向量对应的概率；pClass1:样本数据中类别为1的概率
    p1=sum(vec2Classify*p1Vec)+log(pClass1)    #第一项已经做了log处理
    p0=sum(vec2Classify*p0Vec)+log(pClass1)
    if p1>p0:
        return 1
    else:
        return 0


def testNB():
    list_posts,list_classes=loadDataSet()
    myVocabulary=createVocabList(list_posts)
    print myVocabulary
#     vec=setOfWord2Vec(myVocabulary, list_posts[0])
# #     print vec
    trainMat=[]
    for postinDoc in list_posts:
        vec=setOfWord2Vec(myVocabulary, postinDoc)
        trainMat.append(vec)
    p0V,p1V,pAb=trainNB(trainMat, list_classes)
#     print p0Vpass
#     print p1V
#     print pAb 
    testEnty=['love','my','is']
    thisDoc=array(setOfWord2Vec(myVocabulary, testEnty))
    print testEnty,'classifyed as:',classifyNB(thisDoc, p0V, p1V, pAb)

#将一个单词列表的输入按照vocabList映射成一个向量，其中向量的值为相应单词出现的次数
def bagOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

#将一个长的文件转换成一一个单词列表
def textparse(bigString):
    import  re
    listOfTokens=re.split(r'\W*',bigString)#分隔符是除字母，数字后的任意字符串
    return [tok.lower() for tok in listOfTokens if len(tok)>2]
    
def spamTest():
    docList=[];classList=[];fullText=[];
    for i in range(1,26):#1,2...25分别有25篇spam和ham文档
        wordList=textparse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textparse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=range(50);testSet=[]#样本数据的索引值
    for i in range(10):#随机构造训练数据集
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])#测试数据集
        del(trainingSet[randIndex])#删除这个作为测试的样本的索引，其他的作为训练样本的索引
    trainMat=[];trainClasses=[];
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB(trainMat, trainClasses)
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam)!=classList[docIndex]:
            errorCount+=1
            print "classification error",docList[docIndex]
    print 'the error rate is：',float(errorCount/len(testSet))
    
if __name__=="__main__":
#     testNB()
    spamTest()
    