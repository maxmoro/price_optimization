import sys
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import glob
import textdistance #https://pypi.org/project/textdistance/
from scipy.spatial.distance import euclidean, pdist, squareform
import spacy 

sys.path.append(os.environ['CAPSTONE_DATA'])
sys.path.append(os.environ['CAPSTONE_PYTHON_SOURCE'])

folder = os.environ['CAPSTONE_DATA']
dfPricing = pd.read_pickle(os.path.join(folder,'tidy_data/Pricing_v4/Pricing.pkl'))
   
def distance_funct(u, v):
    return (euclidean(u,v))


def spacy_similarity(w1,w2,nlp):
    doc1 = nlp(w1)
    doc2 = nlp(w2)
    return(doc1.similarity(doc2))
    
def getMatrix(dataIn,periodCol ='WeekDate'):
	tGroup =dataIn.groupby([periodCol,'Product']).agg(Transactions = ('Transactions',sum)).reset_index()
	#Create a Matrix with Values (1= the products been bought together in the same month, 0 = their are not bought in the same month)
	tCrossJoin = pd.merge(tGroup[[periodCol,'Product']], tGroup[[periodCol,'Product']],on=[periodCol],suffixes=['','2'])
	tCrossJoin['Events']=1
	tMatrix = pd.pivot_table(tCrossJoin, values='Events', index=[periodCol,'Product'],columns=['Product2'], aggfunc=sum,fill_value=0) 
	#Summing the matrix by Product (how many times two products have been bought at the same time in the same month)
	tMatrixProduct  = tMatrix.groupby('Product').sum() 
	return(tMatrixProduct)

def calcSimilarity(dataIn,doNLP=False,nlp=None,periodCol='WeekDate'):
    tMatrixProduct=getMatrix(dataIn,  periodCol)
           
    #Calculate the similarity Matrix
    TDist  = pdist(tMatrixProduct, 'euclidean' ) #distance_funct) 
    tSimilMatrix = pd.DataFrame(1/(1+squareform(TDist)), columns=tMatrixProduct.index, index=tMatrixProduct.index)
    
    #converting a DataFrame with Similarity Score and Events Columns
    tSimilDF = pd.melt(tSimilMatrix.reset_index(), id_vars='Product',var_name='Product2', value_name='Similarity')
    tEventsDF = pd.melt(tMatrixProduct.reset_index(), id_vars='Product',var_name='Product2', value_name='Events')
    #add normalized events cols
    tEventsDF['EventsNorm'] = tEventsDF.groupby('Product', group_keys=False).apply(lambda x:x['Events']/max(x['Events']))
    #add Text similairt
    tEventsDF['TextCosine'] = tEventsDF.apply(lambda x:textdistance.cosine.normalized_similarity(x['Product'],x['Product2']),axis=1)
    tEventsDF['TextLev'] = tEventsDF.apply(lambda x:textdistance.levenshtein.normalized_similarity(x['Product'],x['Product2']),axis=1)
    if(doNLP): tEventsDF['TextNLP'] = tEventsDF.apply(lambda x:spacy_similarity(x['Product'],x['Product2'],nlp),axis=1)
    
    tSimilDF=(pd.merge(tSimilDF,tEventsDF,on=['Product','Product2'],how='left')
              .sort_values(by=['Product','EventsNorm','Events','Similarity'],ascending=[True,False,False,False])
              .reset_index(drop=True))
    
    #returning MAtrix and Data Frame
    return(tSimilDF)
	
def loadSimilarity(version=4,allCustomers=False):
    folder = os.environ['CAPSTONE_DATA']
    allC = '_allCustomers' if allCustomers else ''
    dataOut = pd.read_pickle(os.path.join(folder,'tidy_data/Transactions_v' + str(version) + '/Similarity' + allC + '.pkl'))
    return(dataOut)
    

def topSimilar(dfSimilarity,ChainMaster,Product,topn=5,pivot=False,pivotField='Events_NLP',ProductsList=[],exludeSelf=True):
    ProdCat='' 
    if (Product!=''): ProdCat=dfSimilarity[dfSimilarity['Product']==Product]['Category (CatMan)'].iloc[0]
    
    out = dfSimilarity[((dfSimilarity['Chain Master']==ChainMaster) | (ChainMaster=='')) &
                       ((dfSimilarity['Product']==Product) | (Product == '')) & 
                       ((dfSimilarity['Category (CatMan)']==ProdCat) | (Product == '')) &
                       ((dfSimilarity['Product2'].isin(ProductsList)) | (ProductsList == []))  &
                       ((dfSimilarity['Product'] != dfSimilarity['Product2']) | (exludeSelf == False))  
                       #].sort_values(by='Events_NLP',ascending=False).head(topn)
                       ].sort_values(by=pivotField,ascending=False).head(topn)
    if(pivot):
        out=pd.pivot_table(out, values=pivotField, index=['Product'],columns=['Product2'], aggfunc=sum,fill_value=0)
    return(out)


def testTopSimilar():
    dfSimilarity = loadSimilarity(version=4)
    topSimilar(dfSimilarity,ChainMaster='WESTERN BEV LIQ TX',Product='MCCORMICK APPLE VODKA 60  1L',topn=10,pivot=True)
    topSimilar(dfSimilarity,ChainMaster='WESTERN BEV LIQ TX',Product='',topn=10,pivot=True)
    

def getPrice(Product,Day,Qty):
    price=(dfPricing[(dfPricing['Product']==Product) &
                     (dfPricing['Start']<=Day) &
                     (dfPricing['End']>Day) &
                     (dfPricing['Qty']<=Qty)
                     ]
           .sort_values(['Qty'],ascending=False)
           .head(1))
    if(price.shape[0]==0):
         price=(dfPricing[(dfPricing['Product']==Product) &
                     (dfPricing['Qty']<=Qty)
                     ]
           .sort_values(['Start','Qty'],ascending=[True,False])
           .head(1))
    if(price.shape[0]==0): retPrice=np.nan
    else: retPrice=price['Net 9L'].values[0]
    return(retPrice)

def resample(dataIn,periodCol = 'WeekDate',resampleFreq =''):
    if(resampleFreq!=''):
        print("resampling to ",resampleFreq)
        dataR=(dataIn.groupby(['Product','Chain Master',pd.Grouper(key=periodCol,freq=resampleFreq)])
               .apply(lambda x: pd.Series({
                   '9L Cases': sum(x['9L Cases'])
                    ,'Dollar Sales per 9L Case' : (sum(x['9L Cases'] * x['Dollar Sales per 9L Case']) / sum(x['9L Cases']))
                    }))
                .reset_index())
    else:
        print("No Resampling")
        dataR=dataIn
    return(dataR)

def mergeTopSimilar(dataIn,dfSimilarity,ChainMaster,Product,ProductsList,topn=5
                    ,periodCol = 'WeekDate',resampleFreq =''
                    ,encodeCols = True):
    
    pivotCol = 'Dollar Sales per 9L Case'
    
    #RESAMPLING
    dataR =  resample(dataIn=dataIn,periodCol=periodCol,resampleFreq=resampleFreq)

    #Create the target product dataset with the price and quantity for the selected chainmaster
    dataProduct1 = (dataR[((dataR['Chain Master']==ChainMaster) | (ChainMaster == ''))  
                           & (dataR['Product']==Product)
                           ]
                    .groupby([periodCol,'Product'])
                    .apply(lambda x: pd.Series({
                        '9L Cases': sum(x['9L Cases'])
                        ,'Dollar Sales per 9L Case' : (sum(x['9L Cases'] * x['Dollar Sales per 9L Case']) / sum(x['9L Cases']))
                        }))
                    .reset_index())
    
    #adding missing dates
    fullPeriod= pd.DataFrame({periodCol:dataR[periodCol].unique(),'key':0} )
    fullPeriod= fullPeriod[(fullPeriod[periodCol] >= min(dataProduct1[periodCol]))]
    dataProduct1= pd.merge(dataProduct1,fullPeriod,on=periodCol,how='outer').sort_values(by=periodCol).drop('key',axis=1)
    #filling nas
    dataProduct1['Product'].fillna(Product,inplace=True)
    dataProduct1['9L Cases'].fillna(0,inplace=True)
    #find pricing
    dataProduct1['Pricing'] =  dataProduct1.apply(lambda x: getPrice(x['Product'],x['WeekDate'],1),axis=1)
    dataProduct1['Dollar Sales per 9L Case'].fillna(dataProduct1['Pricing'],inplace=True)
    #fill forward
    dataProduct1['Dollar Sales per 9L Case'].fillna( method='ffill',inplace=True)
    dataProduct1['Dollar Sales per 9L Case'].fillna( method='bfill',inplace=True)
    
    #find simialrity
    sim = topSimilar(dfSimilarity=dfSimilarity,ChainMaster=ChainMaster,Product=Product
                     ,ProductsList = ProductsList,topn=topn,pivot=False,exludeSelf=True)
    
    #create products price table for the other products (1,2,3)
    dataProduct2 = (dataR[((dataR['Chain Master']==ChainMaster) | (ChainMaster=='')) &
                           (dataR['Product'].isin(sim['Product2']))]
                    [[periodCol,'Product','Dollar Sales per 9L Case','9L Cases']]
                    .groupby([periodCol,'Product'])
                     .apply(lambda x: pd.Series({
                        'Dollar Sales per 9L Case' : (sum(x['9L Cases'] * x['Dollar Sales per 9L Case']) / sum(x['9L Cases']))
                        }))
                    .reset_index()
                    )
   
    #infer missing prices
    ##from Pricing Table

    #creating full dataset (product and period)    
    tProd = dataProduct2.groupby('Product').agg(**{'DateMin':(periodCol,'min')}).reset_index()
    tProd['key']=0
    fullData = pd.merge(fullPeriod,tProd,on='key',how='outer').drop('key',axis=1)

    #create a Product2 dataset with all the dates for the other products (1,2,3)
    dataProduct2= pd.merge(dataProduct2,fullData,on=[periodCol,'Product'],how='outer').sort_values(by=periodCol)
    #Inferring missing prices
    dataProduct2['Pricing'] =  dataProduct2.apply(lambda x: getPrice(x['Product'],x['WeekDate'],1),axis=1)
    dataProduct2['Dollar Sales per 9L Case'].fillna(dataProduct2['Pricing'],inplace=True)
    #fill forward/backward
    dataProduct2['Dollar Sales per 9L Case'].fillna( method='ffill',inplace=True)
    dataProduct2['Dollar Sales per 9L Case'].fillna( method='bfill',inplace=True)
    
        
    #pivot and merge 
    dataProduct2P = pd.pivot_table(dataProduct2, values=pivotCol, index=[periodCol],columns=['Product'], aggfunc=sum,fill_value=0)
    dataProduct2P  = dataProduct2P[sim['Product2']]
    
    
    dataOut = pd.merge(dataProduct1,dataProduct2P,left_on = periodCol,right_on=periodCol,how='left')
    dataOut=dataOut.drop('Pricing',axis=1)
    colExog = sim['Product2'].tolist()
    encCols =dict()
    decCols =dict()
    if (encodeCols):
        for i,c in enumerate(ProductsList):
            decCols[str(i)]=c
            encCols[c]=str(i)
            if(c in dataOut.columns):
                dataOut.rename(columns={c:str(i)},inplace=True)
        colExog=[encCols[c] for c in colExog]
        dataOut.rename(columns={'Dollar Sales per 9L Case':encCols[Product]},inplace=True)
        colExog = [encCols[Product]] + colExog
    else:
        colExog = ['Dollar Sales per 9L Case'] + colExog
    return(dataOut,colExog,encCols,decCols)
      
def testMergeTopSimiar():
    from ETL.ETL import loadDataset
    from ETL.ETL import getTopProducts
    dataIn=loadDataset(version=4)
    dfSimilarity = loadSimilarity(version=4)
    dfSimilarityAll = loadSimilarity(version=4,allCustomers=True)
    ChainMaster = '' #'SPECS'
    ProdCat='SUP PREM WHISKEY'
    ProductsList = getTopProducts(dataIn,ChainMaster=ChainMaster,ProdCat=ProdCat,topN=3)
    Product = ProductsList[2]
    t= mergeTopSimilar(dataIn,dfSimilarity=dfSimilarityAll
                    ,ChainMaster=ChainMaster
                    ,Product=Product
                    ,ProductsList =ProductsList
                    ,topn=5)
    t[0].hist()
    t2=mergeTopSimilar(dataIn,dfSimilarityAll
                    ,ChainMaster=ChainMaster
                    ,Product=Product
                    ,ProductsList =ProductsList
                    ,resampleFreq='M'
                    ,topn=5)
    t2[0].hist()
    prodEncoder = dataIn.groupby(['Product'])['Product ID'].first().to_dict()
    getHeatmap(dfSimilarityAll,ChainMaster = ChainMaster,prodCat = ProdCat,size=15,norm=True,encoder=prodEncoder,prodsList = ProductsList)
    #Product='GENTLEMAN JACK WHSKY OL 750M'

    
<<<<<<< HEAD
def getHeatmap(dfSimilarity,ChainMaster,prodCat,size=20,cmap='YlGn',norm=False,encoder=dict(),prodsList = []
               ):
    
    data=dfSimilarity[((dfSimilarity['Chain Master']==ChainMaster ) | (ChainMaster=='')) & (dfSimilarity['Category (CatMan)'] == prodCat)]
   
    
=======
def getHeatmap(dfSimilarity,ChainMaster,prodCat,size=20,cmap='YlGn',norm=False,encoder=dict(),prodsList = []):
    
    data=dfSimilarity[((dfSimilarity['Chain Master']==ChainMaster ) | (ChainMaster=='')) & (dfSimilarity['Category (CatMan)'] == prodCat)]
>>>>>>> 9aa037ce19e03e3e82b6ee96ab4dbcfdf52e6819
    if len(encoder)!=0: 
        data=data.replace({'Product':encoder,'Product2':encoder})
        prodsList = [encoder[p] for p in prodsList]
        prodsList.sort
    if len(prodsList) !=0: 
        data=data[(data['Product'].isin (prodsList)) & (data['Product2'].isin (prodsList))]
        data.reindex(prodsList)
<<<<<<< HEAD
    mtrx = pd.pivot_table(data, values='Events_NLP', index=['Product'],columns=['Product2'], aggfunc=sum,fill_value=0) 
=======
    mtrx = pd.pivot_table(data, values='TextNLP', index=['Product'],columns=['Product2'], aggfunc=sum,fill_value=0) 
>>>>>>> 9aa037ce19e03e3e82b6ee96ab4dbcfdf52e6819
    #mtrx = getMatrix(data)
    if(norm):
        mtrx.loc[:,:] = mtrx.loc[:,:].div(mtrx.max(axis=0), axis=0)
    if len(prodsList) !=0:
        order=  [np.where(mtrx.columns ==x)[0][0] for x in prodsList]
    else:
        order=np.argsort(-mtrx.sum().to_numpy())
    mt=mtrx.iloc[order,order]
    size=min(size,mt.shape[0])
    heatmap=sns.heatmap(mt.iloc[range(size),range(size)],cmap=cmap)
    return(heatmap,mt)

def testGetHeatmap():
    from similarity.similarity import getHeatmap
    from similarity.similarity import loadSimilarity
    from ETL.ETL import getTopProducts
    from ETL.ETL import loadDataset
    dfSimilarity = loadSimilarity(version=3)
    dataRaw= loadDataset(version=4)
    custs= ['WESTERN BEV LIQ TX','SPECS','INDEPENDENTS']# dfSimilarity['Chain Master'].unique()
    maxc =len(custs)
    maxp = dfSimilarity['Category (CatMan)'].nunique()
    plt.figure(figsize=(20,10*maxc*maxp))
    prodEncoder = dataRaw.groupby(['Product'])['Product ID'].first().to_dict()
    for ic,c in enumerate(custs):
        prods = dfSimilarity[dfSimilarity['Chain Master']==c]['Category (CatMan)'].unique()
        for ip,p in enumerate(prods):
            plt.subplot(maxc*maxp,1,(ic)*maxp+(ip)+1)
            prodsList = getTopProducts(dataRaw,ChainMaster = c, ProdCat = p,topN=4)
            ax=getHeatmap(dfSimilarity,ChainMaster = c,prodCat = p,size=15,norm=True,encoder=prodEncoder,prodsList = prodsList)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=16)
            plt.title('Chain: ' + c +'  Product: ' + p,size=20)
            plt.tick_params(axis='y',labelsize=14)
            plt.tick_params(axis='x',labelsize=14)
            plt.xlabel('')
            plt.ylabel('')
    plt.tight_layout()
#testGetHeatmap()