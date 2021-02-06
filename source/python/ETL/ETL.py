import sys
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
sys.path.append(os.environ['CAPSTONE_DATA'])

def loadDataset(version=4):
    folder = os.environ['CAPSTONE_DATA']
    t=pd.read_pickle(os.path.join(folder,'tidy_data/Transactions_v'+ str(version) + '/Transactions.pkl'))
    dataRaw=t[t['Category (CatMan)'].isin(['SUP PREM WHISKEY','ECONOMY VODKA'])]
    return(dataRaw)
    
def getTopProducts(dataIn,ChainMaster,ProdCat,field='Dollar Sales',topN=10,timeCol = 'WeekDate'):
    
    t=(dataIn[((dataIn['Chain Master']==ChainMaster) | (ChainMaster=='') ) & (dataIn['Category (CatMan)']==ProdCat)]
       .groupby(['Product',timeCol])
       .apply(lambda x: pd.Series({
                    'sum': sum(x[field])
                    }))
       .reset_index()
       .groupby('Product')['sum']
       .agg(['mean','count'])
       .sort_values(by='mean',ascending=False)
       .reset_index()
       )
    #70 is a good cut for the min number of periods for a product, but if we don't have enough products, we need to lower the cut
    bestCount = t['count'].sort_values(ascending=False).iloc[topN]
    if(bestCount >70): bestCount =70
    t2=t.query('count >= ' + str(bestCount))
    prods=t2.Product.head(topN).to_list()
    return(prods) #,dataIn[dataIn['Product'].isin(prods)])
    
def testgetTopProducts():
    dataIn=loadDataset(version=4)
    getTopProducts(dataIn,ChainMaster='',ProdCat='SUP PREM WHISKEY')
    

    
    
    