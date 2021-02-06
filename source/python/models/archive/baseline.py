# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:07:00 2020

@author: annam
"""
import numpy as np
import math

def baselinePred(dataModel,colTime = 'WeekDate'
                 ,trainPeriods = 6
                 ,testPeriods = 3
                 ,windowStart= -9
                 ,predictionPeriods=3
                 #,priceCol = '0'
                 ,qtyCol = '9L Cases'):
    
    dates = np.sort(dataModel[colTime].unique())
    trainDates = dates[(len(dates)+windowStart):(len(dates)+windowStart+trainPeriods)]
    testDates = dates[(len(dates)+windowStart+trainPeriods):(len(dates)+windowStart+trainPeriods+testPeriods)]
    
    #TEST Baseline Model
    train= dataModel[dataModel[colTime].isin(trainDates)]
    test= dataModel[dataModel[colTime].isin(testDates)]
    qtyTestPred =  np.mean(train[qtyCol] )
    qtyTestRMSE=  math.sqrt( sum((qtyTestPred-test[qtyCol])**2))
    #revenueTestPred= np.mean(train[priceCol] * train[qtyCol])
    #revenueTestRMSE =  math.sqrt( sum((revenueTestPred-(test[priceCol]*test[qtyCol]))**2))
    
    #Pred Baseline
    trainDates = dates[(len(dates)-trainPeriods):]
    train= dataModel[dataModel[colTime].isin(trainDates)]
    qtyPred= np.mean(train[qtyCol] )
    #revenuePred= np.mean(train[priceCol] * train[qtyCol] )
    pred = np.repeat(qtyPred,predictionPeriods)
    #pred = np.repeat(revenuePred,predictionPeriods)

    return(pred,qtyTestRMSE)
    
    

def testBaseLine():
    from ETL.ETL import loadDataset, getTopProducts
    from similarity.similarity import mergeTopSimilar, loadSimilarity
    dataIn= loadDataset(version=4)
    dfSimilarity = loadSimilarity(version=4)
    ChainMaster = 'SPECS'
    ProdCat='SUP PREM WHISKEY'
    prods = getTopProducts(dataIn,ChainMaster=ChainMaster,ProdCat=ProdCat,topN=5)
    prods
    prod = prods[0]
    colTime = 'WeekDate'
    (dataModel,_,_,_) = mergeTopSimilar(dataIn,dfSimilarity
                    ,ChainMaster=ChainMaster
                    ,Product=prod
                    ,ProductsList =prods
                    ,topn=5)
    bl = baselinePred(dataModel,colTime = colTime
                 ,trainPeriods =6,testPeriods = 3,windowStart= -9
                 #,seasonPeriod = 12, seasonsCount = 2                 
                 ,predictionPeriods=3 #rolling 
                 ,qtyCol = '9L Cases')
    print(bl)
    