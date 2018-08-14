#coding=utf-8
'''
Created on Nov 20, 2015

@author: zxf
'''
from numpy import *
import operator
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]#交易记录的列表

#Apriori算法：C1-L1-C2-L2:这里就是为了产生C1，就是所有的商品项的类别
def createC1(dataSet):
    c1=[]   #用[[1],[2]...]来表示每一个商品项，因为后面会有集合操作：C2:[[1,2],[2,3]...]
    for transaction in dataSet:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])
    c1.sort()
    return map(frozenset, c1)#将c1:[[1],[2]...]转变成不可改变的set,这样才能在后面作为(ssCnt={}#key:Ck中的每一项，value:频次)的key来使用

#根据初始数据。计算Ck中每一项的支持度，返回满足最小支持度的项
def scanD(D,Ck,minSupport):
    ssCnt={}    #key:Ck中的每一项，value:频次
    #赋值ssCnt
    for can in Ck:#这两层循环可以颠倒位置,效率是一样的
        for tid in D:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
#     print ssCnt #没有按照Ck中的顺序插入到key:{frozenset([4]): 1, frozenset([5]): 3, frozenset([2]): 3, frozenset([3]): 3, frozenset([1]): 2}
    numItems=len(D)#所有交易记录的总数
    retList=[]      #待返回的满足最小支持度的Ck
    supportData={}#key:Ck中的每一项，对应的支持度
    for key in ssCnt:
        support=ssCnt[key]/float(numItems)#支持度，用比率的形式来表示,提前将数据变成float，免得得到的全是0
#         print support
        if support >=minSupport:
            retList.insert(0, key)
        supportData[key]=support
    return retList,supportData

#根据Lk,以及Ckz中项集元素的个数，得到Ck
def aprioriGen(Lk,k):
    retList=[]#待返回的Ck
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2];L2=list(Lk[j])[:k-2]#Lk[[1,2],[3,4],[1,2]]:k=3的时候，对[1,2]而言：比较后面的所有的Lk[j][3-2]是否相等
            L1.sort();L2.sort();
            if L1==L2:
                retList.append(Lk[i]|Lk[j])#如果相等，则用并集操作
    return retList
    
def apriori(dataSet,minSupport=0.5):
    C1=createC1(dataSet);
    D=map(set,dataSet)
    L1,supportData=scanD(D, C1,  minSupport)
    L=[L1]
    k=2
    while len(L[k-2])>0:
        Ck=aprioriGen(L[k-2], k)
        Lk,supK=scanD(D, Ck, minSupport)
        supportData.update(supK)#增加指定字典数据到字典
        L.append(Lk)
        k+=1
    return L,supportData
    
if __name__ == '__main__':
    dataSet=loadDataSet()
    print dataSet
#     c1=createC1(dataSet)
#     print c1
#     D=map(set,dataSet)#这行代码把dataSet中的元素，分别变成set[set([1, 3, 4]), set([2, 3, 5]), set([1, 2, 3, 5]), set([2, 5]), set([1, 3, 4])]
#     print D 
#     L1,supportData0=scanD(D, c1, 0.5)
#     print L1
    L,suppData=apriori(dataSet)
    print L
    print suppData