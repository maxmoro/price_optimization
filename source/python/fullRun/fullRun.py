# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:35:28 2020

@author: max moro
"""
try: 
    import os
    # import glob
    # import sys
    import math
    from typing import List, Optional
    from functools import partial
    import itertools
    import copy
    import datetime
except Exception as e:
    print(e)
    print("Some of the libraries needed to run this script were not installed or were not loaded. Please install the libraries before proceeding.")
    
try:
    # Data Tables
    import pandas as pd # type: ignore
    import numpy as np # type: ignore

    # Plotting
    import matplotlib.pyplot as plt # type: ignore
    import plotly.offline as py # type: ignore
    from plotly.offline import plot # type: ignore
    py.init_notebook_mode(connected=True)

    # EDA and Feature Engineering
    from scipy.spatial.distance import euclidean, pdist, squareform # type: ignore
    import statsmodels.api as sm # type: ignore

    # Parallel Processing
    from joblib import Parallel, delayed # type: ignore

    # Auto Time Series
    from tscv import GapWalkForward # type: ignore
    from sktime.forecasting.naive import NaiveForecaster # type: ignore
    import auto_ts as AT

    # Optimizer
    from skopt import gp_minimize # type: ignore
    from skopt.space import Real, Integer # type: ignore
    from skopt.plots import plot_convergence # type: ignore
    from skopt.plots import plot_objective, plot_histogram

    # Writing Results
    import xlsxwriter # type: ignore
    
    # Local Libraries
    from ETL.ETL import getTopProducts, loadDataset
    from similarity.similarity import mergeTopSimilar, loadSimilarity
    from charting.charting import surface3DChart
except Exception as e:
    print(e)
    print("Some of the libraries needed to run this script were not installed or were not loaded. Please install the libraries before proceeding.")

TOP_PRODUCTS = 0  # How many products to consider in the category
TOP_SIMILAR = 0   # Get TOP_SIMILAR most similar products
RUN_PERIODS: List[int] = [] # this will run AutoTS and Optimzer for speicfic periods (1 = last, 2 = second to last)

CV_NUM = 0         # number of Cross Valudations  10
OPT_CALLS = 0     # number of calls for the optimizer process   100
OPT_RND_STARTS = 0 # number of random calls for the optimizer process  20

LOG_TRANSFORM = True # Take log of 9L cases to smooth out peaks and valleys
ZERO_ADDER = 0

RESAMPLE_FREQ = ''
FORECAST_PERIOD = 0
SEASONAL_PERIOD = 0
COL_TIME = 'WeekDate'
COL_PREDS = ['9L Cases'] #Demand
COL_PRICE= ['Dollar Sales per 9L Case'] #Price

outputColsModels =['Chain Master','Product','Period to Last','White Noise'
                   ,'Naive Best Type','Naive Best RMSE'
                   ,'P0 Best Model Name','P0 Best Model RMSE'
                   ,'P0+Sim Best Model Name','P0+Sim Best Model RMSE'
                   ]
outputColsOptP0 = ['Chain Master','Product','Period to Last'
                   ,'P0 Non-Opt Price','P0 Non-Opt Demand','P0 Non-Opt Revenue','P0 Non-Opt Chain Master Revenue'
                   ,'P0 Optimal Price','P0 Demand','P0 Revenue','P0 Chain Master Revenue'
                   ]

outputColsOptP0Sim = ['Chain Master','Product','Period to Last'
                      ,'P0+Sim Non-Opt Price','P0+Sim Non-Opt Demand','P0+Sim Non-Opt Revenue','P0+Sim Non-Opt Chain Master Revenue'
                      ,'P0+Sim Optimal Price','P0+Sim Demand','P0+Sim Revenue','P0+Sim Chain Master Revenue'
                      ]
outputColsOptSummary = ['Chain Master','Product'
                        ,'P0 Non-Opt Revenue','P0 Opt Revenue','P0 Opt Diff','P0 Opt Diff Ratio'
                        ,'P0+Sim Non-Opt Revenue','	P0+Sim Opt Revenue','P0+Sim Opt Diff','P0+Sim Opt Diff Ratio'
                        ]
outputColsOptNaive = ['Chain Master','Product','Period to Last'
                      ,'Actual Price','Actual Demand','Actual Revenue','Actual Chain Master Revenue'
                      ,'Naive Prices','Naive Demand','Naive Revenue','Naive Chain Master Revenue'
                      ]
outputColsOptAll = list(dict.fromkeys(outputColsOptNaive+outputColsOptP0+outputColsOptP0Sim))
#################
#### AUTO TS ####
#################
  
def modelsLoadData(ProductsList,dataRaw,ChainMaster,leaveOutLastPeriods=0):
    all_data = []
    
    if(ChainMaster!=''):
        dfSimilarity = loadSimilarity(version=4)
    else:
        dfSimilarity = loadSimilarity(version=4,allCustomers=True)
    
    for i, Product in enumerate(ProductsList):
        (dataModel,colExog,colEnc,colDec) = mergeTopSimilar(dataRaw, dfSimilarity
                                                            ,ChainMaster=ChainMaster
                                                            ,Product=Product
                                                            ,ProductsList=ProductsList
                                                            ,topn=TOP_SIMILAR 
                                                            ,periodCol = COL_TIME
                                                            ,resampleFreq=RESAMPLE_FREQ
                                                            ,encodeCols=True)

        if(leaveOutLastPeriods>0):
            periods = dataModel[COL_TIME].unique()
            periods.sort()
            keepPeriods = periods[:-leaveOutLastPeriods]
            dataModel=dataModel[dataModel[COL_TIME].isin(keepPeriods)]
        
        if i == 0: print(f"Decoder: {colDec}")

        print("\n\n")
        print("-"*50)
        print(f"Product: {colDec.get(str(i))}")
        print("-"*50)

        #colExog = colExog + colEndog
        print(f"Exogenous Price Columns: {colExog}")

        allCols=[COL_TIME]+COL_PREDS+ colExog
        data=dataModel[allCols]
        print(f"% of weeks without a purchase: {sum(data['9L Cases'] == 0)/data.shape[0]*100}")
        all_data.append(data)
    
    all_data_non_transformed =  copy.deepcopy(all_data)
    
    if LOG_TRANSFORM: 
        print("Log Transforming")
        for i in np.arange(len(all_data)):
            all_data_non_transformed[i] = all_data[i].copy(deep=True)
            all_data[i][COL_PREDS] = np.log10(all_data[i][COL_PREDS] + ZERO_ADDER)
            print(f"\tProduct: {colDec.get(str(i))}")
    return(all_data,all_data_non_transformed,colExog,colEnc,colDec)
            
def ModelsWhiteNoise(all_data)           :
    ## WHITE NOISE TEST
    white_noise_all = []
    white_noise_df_all = []
    #check if there are 12, 24, 48 data points
    for i, data in enumerate(all_data):
        lags=[12,24,48]
        lags=[x  for x in lags if x < data.shape[0]]
        white_noise_df = sm.stats.acorr_ljungbox(data[COL_PREDS], lags=lags, return_df=True)
        white_noise_df_all.append(white_noise_df)
        if any(white_noise_df['lb_pvalue'] > 0.05):
            white_noise = True
        else:
            white_noise = False
        white_noise_all.append(white_noise)

        print(white_noise_df)
        print(f"\nIs Data White Noise: {white_noise}")
    
    return(white_noise_all)  

def ModelsTestTrain(all_data,all_data_non_transformed):
    all_train = []
    all_test = []

    all_train_non_transformed = []
    all_test_non_transformed = []

    for i, data in enumerate(all_data):
        train = all_data_non_transformed[i].iloc[:-FORECAST_PERIOD]
        test = all_data_non_transformed[i].iloc[-FORECAST_PERIOD:]
        all_train_non_transformed.append(train)
        all_test_non_transformed.append(test)

        train = data.iloc[:-FORECAST_PERIOD]
        test = data.iloc[-FORECAST_PERIOD:]
        all_train.append(train)
        all_test.append(test)

        print(train.shape,test.shape)
    return(all_train,all_test,all_train_non_transformed,all_test_non_transformed)

def ModelsFit(all_data,all_train,all_test,withSimilar,model_type=['SARIMAX','ML','prophet','auto_SARIMAX']):
        
    def modelsFun(i):
        train = all_train[i]
        test = all_test[i]
        if(withSimilar==False):
            train = train[train.columns[0:3]] #3rd col has the curr product price
        print(train.columns)

        automl_model = AT.AutoTimeSeries(
            score_type='rmse', forecast_period=FORECAST_PERIOD, # time_interval='Week',
            non_seasonal_pdq=None, seasonality=True, seasonal_period=SEASONAL_PERIOD,
            model_type=model_type,
            verbose=0)
        
        #colP = COL_PREDS[COL_PREDS in train.columns]
        automl_model.fit(train, COL_TIME, COL_PREDS, cv=CV_NUM, sep=',') #cv=10
        return(automl_model)
    
    args = np.arange(len(all_data))
    
    all_models = Parallel(n_jobs=-1, verbose=1
                          #, backend="threading"
                           , backend="loky"
                         )(
             map(delayed(modelsFun), args))
    
    
    return(all_models)

def get_rmse(predictions, targets):
    return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())

def modelNaive(all_data,all_train,all_test,all_train_non_transformed,season=12,windowLength=8):
    # import statistics 
    all_naives=pd.DataFrame(columns=['ID','Best Type','Best RMSE'])
    types=['last','seasonal_last','mean']
    #add window code
    
    NFOLDS=5
    for i, data in enumerate(all_data):
        yTrain = pd.Series(all_train[i][COL_PREDS[0]])
        yTest = pd.Series(all_test[i][COL_PREDS[0]])
        yTrain = yTrain.append(yTest) # merging as we are gong to do cv
        rmses=[]
        naive_models=[]
        for t in types:
            #naive_forecaster = NaiveForecaster(strategy="last")
            cv = GapWalkForward(n_splits=10, gap_size=0, test_size=FORECAST_PERIOD)
            cvRmse=[]
            for fold_number, (train, test) in enumerate(cv.split(yTrain)):
                cv_train = yTrain.iloc[train]
                cv_test = yTrain.iloc[test]
                
                naive_forecaster = NaiveForecaster(strategy=t,sp=season,window_length=windowLength)
                naive_forecaster.fit(cv_train)
                yPred = naive_forecaster.predict(np.arange(len(cv_test)))
                rmse=get_rmse(yPred, cv_test)
                cvRmse.append(rmse)
            #naive_models.append(naive_forecaster) #last forecaster
            rmses.append(np.mean(cvRmse))
        bestRmse = np.argmin(rmses)
        bestModel = NaiveForecaster(strategy=types[bestRmse],sp=season)
        yTrainNonTrasformed = pd.Series(all_train_non_transformed[i][COL_PREDS[0]]) 
        bestModel.fit(yTrainNonTrasformed)
        all_naives=all_naives.append(
            {'ID':i
             ,'Best Type': types[bestRmse]
             ,'Best RMSE': rmses[bestRmse]
             ,'Best Naive': bestModel
             ,'All Types': [types]
             ,'All RMSEs': [rmses]
             ,'All Naives':naive_models
            }
            ,ignore_index=True)
    print(all_naives)
    return(all_naives) 

def centerLog(text,w,pre='\n',post=''):
    t=int((w-len(text))/2-1)
    return(pre+'='*t+' '+text+' '+'='*(w-len(text)-t-2)+post)

def printLog(main,subs,period = 1,linesPre=2,linesPost=1):
    if(isinstance(subs,list)== False): subs=[subs]
    maxw=max([len(x) for x in [main] + subs])+10
    print("\n"*linesPre
          +"="*maxw+" ("+str(datetime.datetime.now())+")"
          +centerLog(main,maxw)
          +''.join([centerLog(x,maxw) for x in subs])
          +"\n"+"="*maxw
          +"\n"*linesPost
         )
  

def runModels(ProductsList,dataRaw,ChainMaster,leaveOutLastPeriods=0):
    
    printLog("GET DATA",ChainMaster)
    all_data,all_data_non_transformed,colExog,colEnc,colDec = modelsLoadData(ProductsList
                                                                             ,dataRaw
                                                                             ,ChainMaster
                                                                             ,leaveOutLastPeriods=leaveOutLastPeriods)
    
    printLog("WHITE NOISE",ChainMaster)
    white_noise = ModelsWhiteNoise(all_data)  
    
    printLog("TEST/TRAIN",ChainMaster)
    all_train, all_test,all_train_non_transformed,all_test_non_transformed = ModelsTestTrain(all_data,all_data_non_transformed)
    
    modelsStats = pd.DataFrame()
    modelsStats['Product'] = ProductsList
    modelsStats['Chain Master'] = ChainMaster
    modelsStats['White Noise'] = white_noise
    
    printLog("NAIVE",ChainMaster)
    naive = modelNaive(all_data,all_train,all_test,all_data_non_transformed,season=4,windowLength=8)
    modelsStats['Naive Best Type'] = [naive.iloc[x]['Best Type'] for x in np.arange(len(all_data)) ]
    modelsStats['Naive Best RMSE'] = [naive.iloc[x]['Best RMSE'] for x in np.arange(len(all_data)) ]
    modelsStats['Naive Best Model'] = [naive.iloc[x]['Best Naive'] for x in np.arange(len(all_data)) ] 
     
    printLog("Multivar P0",ChainMaster)
    multivarP0 = ModelsFit(all_data,all_train,all_test,withSimilar = False)
    modelsStats['P0 Best Model Name'] = [multivarP0[x].get_leaderboard().iloc[0]['name'] for x in np.arange(len(all_data)) ]
    modelsStats['P0 Best Model RMSE'] = [multivarP0[x].get_leaderboard().iloc[0]['rmse'] for x in np.arange(len(all_data)) ]
    modelsStats['P0 Best Model'] = multivarP0 #[multivarP0[x] for x in np.arange(len(all_data)) ]
    
    printLog("Multivar P0+Sim",ChainMaster)
    multivarP0Sim = ModelsFit(all_data,all_train,all_test,withSimilar = True )
    modelsStats['P0+Sim Best Model Name'] = [multivarP0Sim[x].get_leaderboard().iloc[0]['name'] for x in np.arange(len(all_data)) ]
    modelsStats['P0+Sim Best Model RMSE'] = [multivarP0Sim[x].get_leaderboard().iloc[0]['rmse'] for x in np.arange(len(all_data)) ]
    modelsStats['P0+Sim Best Model'] = multivarP0Sim #[multivarP0Sim[x] for x in np.arange(len(all_data)) ]
   
    printLog("COMPLETED",ChainMaster)
    return(modelsStats)

###################
#### OPTIMIZER ####
###################
    
def complex_objective(x: List
                      , ts_index_name: str
                      , ts_index: List
                      , all_models: List
                      , all_data: List
                      , mask: Optional[List[bool]] = None
                      , verbose: int = 0
                      , return_individual: bool = False
                      , logT = False
                      , P0_only = False
                      ):
    """
    :param x A list of product pricing for which the revenue has to be computed
    :type x List
    :param mask: If the customer is not going to purchase a product in a period, we can choose to omit it from the revenue calculation in the optimizer.
                 Default = None (considers all products in revenue calculation)
    :type mask  Optional[List[bool]]

    :param ts_index The index to use for the test data. This is needed for some models (such as ML) that use this to create features
    :type ts_index List

    :param return_individual If True, this returns the individual revenue values as well
                             Used mainly when this function is called standalone. Set of False for optimization
    :type return_individual bool

    :param verbose Level of verbosity (Default: 0). This is set to 1 or 2 (mainly for debug purposes)
    :type verbose int
    """
    if verbose >0: print ("### Prediction Function ###")
    # Create test data from input
    index = [str(i) for i in np.arange(len(x))]
    x_df = pd.DataFrame(x, index = index)
    x_df = x_df.T

    # Set index (important for some models)
    x_df.index = ts_index[0:1]
    x_df.index.name = ts_index_name

    # If mask is not provided, use all
    if mask is None:
        mask = [False for item in x] 

    if verbose >= 2:
        print(x_df.info())
        print(x_df.columns)

    total_revenue = 0
    revenue = []
    

    for i in np.arange(len(all_data)):
        if verbose >= 1:
            print("\n" + "-"*50)
            print(f"Product Index: {i}")
        
        if not mask[i]:
            if P0_only: columns = [all_data[i].columns[-(TOP_SIMILAR+1)]]
            else: columns = all_data[i].columns[-(TOP_SIMILAR+1):].values #columns[-(TOP_SIMILAR+2)] for the P0 only type
            if verbose >= 2:
                print(f"All Columns in Test Data: {columns}")
                print('i:',i)
                print(x_df[columns])
                print("----------------------------------")

            test_data = x_df[columns]
            prediction = all_models[i].predict(X_exogen = test_data,forecast_period=1) #change this back when Nikhil fixes the autoTS
            
            if verbose >= 2: print(f"Prediction Type: {type(prediction)}")
            if verbose >= 1: print(f"Demand Prediction (transformed): {prediction}")

            # If model was created with log transformation
            if logT:
                prediction = 10**prediction
                if verbose >= 1:
                    print("\nDemand Prediction (Original)")
                    print(prediction)
                
            product_revenue = prediction * x[i]

            # TODO: Clamping - Fix later (this gives an error with pandas. We need to pluck it out as a value)
            # product_revenue = max(product_revenue, 0)  # Clamp at min value of 0 for predictions that are negative

            if verbose >= 1: print(f"Product Revenue: ${round(product_revenue)}")
                                
            if isinstance(product_revenue, pd.Series):
                product_revenue = product_revenue.iloc[0]
            revenue.append(product_revenue)
                
            # total_revenue = total_revenue + product_revenue
        else:
            if verbose >= 1: print("This product's revenue was not included since it was not ordered by the customer in this period.")
            product_revenue = 0
            revenue.append(product_revenue)

        if verbose >= 1: print("-"*50 + "\n")
        
    total_revenue = sum(revenue)

    if verbose >= 1:
        print("\n\n" + "="*50)
        print(f"Total Revenue: ${round(total_revenue)}")
        print("="*50 + "\n\n")
        print ("### Prediction Function END ###")
    if return_individual is True: return -total_revenue, revenue      
    
    return -total_revenue
    

def opt_get_mask(all_data,all_test):
    # Did the customer actually want to but products in that period?
    # Only include the revenue in the objective if they actually ordered it
    # This model is not trying to predict if they would purchase a product when they were not going to purchase it earlier.
    # That requires a lot of human psychology and may not be captured in the model

    INCLUDE_MASKING = True

    mask: List[bool] = []
    for index in np.arange(len(all_data)):
        if INCLUDE_MASKING:
            if all_test[index].iloc[0]['9L Cases'] == 0:
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(False)

    print(f"Mask: {mask}")
    return(mask)

def opt_get_space(all_data,MARGIN=0.0):
    MARGIN = 0.0 # How much to go over or under the min and max price respectively during the search for optimial revenue
    space = []

    for index in np.arange(len(all_data)):
        #min_val = all_data[index][str(index)].min()
        min_val = np.percentile(all_data[index][str(index)], 10)
        #max_val = all_data[index][str(index)].max()
        max_val = np.percentile(all_data[index][str(index)], 90)
        min_limit = min_val*(1-MARGIN)
        max_limit = max_val*(1+MARGIN)
        if min_limit >= max_limit: 
            print("Min Limit is >= than Max Limit, setting them near median (+-5%)")
            min_limit=np.percentile(all_data[index][str(index)], 50)*0.95
            max_limit=np.percentile(all_data[index][str(index)], 50)*1.05
        space.append(Real(low=min_limit, high=max_limit, prior='uniform'))

    return(space)

def opt_get_func(all_data,all_models,complex_objective,test_index_name,test_index,mask,verbose=0,P0_only=False):
    
    # create a new function with mask
    masked_complex_objective = partial(complex_objective
                                       ,ts_index_name=test_index_name
                                       ,ts_index=test_index
                                       ,mask=mask
                                       ,logT=LOG_TRANSFORM,verbose=verbose
                                       ,all_models=all_models
                                       ,all_data=all_data
                                       ,P0_only=P0_only)
    if False:
        if P0_only:
            print(f"Revenue P0: ${-round(complex_objective([266.51, 195.06, 205.3], ts_index_name=test_index_name, ts_index=test_index, mask=mask,logT=LOG_TRANSFORM,verbose=verbose,all_models=all_models,all_data=all_data,P0_only=True))}")    
        else:
            print(f"Revenue without masking: ${-round(complex_objective([266.51, 195.06, 205.3], ts_index_name=test_index_name, ts_index=test_index, logT=LOG_TRANSFORM,verbose=verbose,all_models=all_models,all_data=all_data))}")
            print(f"Revenue with masking: ${-round(masked_complex_objective([266.51, 195.06, 205.3],verbose=verbose,all_models=all_models,all_data=all_data))}")
    return(masked_complex_objective)

def opt_optimize(masked_complex_objective,space,ChainMaster):
    out=pd.DataFrame()
    P0_only = masked_complex_objective.keywords['P0_only']
    txt = 'P0' if P0_only else 'P0+Sim'
    res = gp_minimize(masked_complex_objective
                      ,space
                      ,acq_func="EI"
                      ,n_calls=OPT_CALLS
                      ,n_random_starts=OPT_RND_STARTS
                      ,random_state=42)    

    ##plotting the dependance plot
    #_ = plot_objective(res.optimizer_results_[0],size=4)
    #plt.show()
    
    ## GET OUTPUT DATA ##
    printLog("OUTPUT P0",ChainMaster)
    out[txt+' Optimal Price'] = [round(price, 2) for price in res.x]
    out[txt+' Chain Master Revenue'] = round(-res.fun)
    _,all_revenues =  masked_complex_objective(res.x, return_individual=True)
    out[txt+' Demand'] = [round(x) for x in (np.array(all_revenues) / np.array(out[txt+' Optimal Price'])).tolist()]
    out[txt+' Revenue'] = [round(x) for x in all_revenues]

    #out['total_test_data_revenue_'+txt] = opt_get_data(all_data,all_test_non_transformed)
    return(out)

def opt_get_non_optimized(masked_complex_objective,all_test_non_transformed,ChainMaster):
    P0_only = masked_complex_objective.keywords['P0_only']
    prices=[all_test_non_transformed[index][str(index)].item() for index in np.arange(len(all_test_non_transformed)) ]
    pred=masked_complex_objective(x=prices,return_individual=True)
    ## GET OUTPUT DATA ##
    out=pd.DataFrame()
    txt = 'P0' if P0_only else 'P0+Sim'
 
    printLog("OUTPUT P0",ChainMaster)
    out[txt+' Non-Opt Price'] = [round(price, 2) for price in prices]
    out[txt+' Non-Opt Chain Master Revenue'] = round(-pred[0])
    out[txt+' Non-Opt Demand'] = [round(x) for x in np.array(pred[1]) / np.array(out[txt+' Non-Opt Price']).tolist()]
    out[txt+' Non-Opt Revenue'] = [round(x) for x in pred[1]]

    return(out)
    

def opt_get_data(all_data,all_test_non_transformed):
    total_test_data_revenue = 0
    for index in np.arange(len(all_data)):
        product_price = all_test_non_transformed[index].iloc[0][str(index)]
        product_demand = all_test_non_transformed[index].iloc[0]['9L Cases']
        product_revenue = product_price * product_demand
        print(f"Product {index} Price 9L Case: ${round(product_price,2)} Revenue: ${round(product_revenue)}")
        total_test_data_revenue = total_test_data_revenue + product_revenue

    print(f"Total Revenue: ${round(total_test_data_revenue)}")
    return(total_test_data_revenue)

def opt_naive(all_models,all_test_non_transformed):
    #uses test price and predict demand based on naive model
    product_price=[]
    product_demand=[]
    product_revenue=[]
    for index in np.arange(len(all_models)): 
        product_price.append(all_test_non_transformed[index].iloc[0][str(index)])
        product_demand.append(all_models[index].predict([0]).tolist()[0])
        product_revenue.append(product_price[index] * product_demand[index])
    total_revenue = sum(product_revenue)
    return(product_price,product_demand,product_revenue,total_revenue)
    
def opt_get_chart(all_data,all_models,space,ChainMaster,ProdCat,test_index,test_index_name,verbose=1,STEPS=5,displayPlots=True,savePath = '3d_charts/',period=0):
    math.ceil(space[0].low)
    math.floor(space[0].high)
    xs = np.arange(math.ceil(space[0].low), math.floor(space[0].high), step=5)
    ys = np.arange(math.ceil(space[1].low), math.floor(space[1].high), step=5)

    allp = [np.arange(math.ceil(space[i].low), math.floor(space[i].high), step=STEPS) for i in np.arange(len(all_data))] 

    if verbose >= 1:
        print("-"*100)
        print(f"Price intervals for product 0: {allp[0]}")
        print(f"Price intervals for product 1: {allp[1]}")
        print(f"Price intervals for product 2: {allp[2]}")
        print("-"*100, "\n")
    filenames=[]
    for i in np.arange(len(all_data)):
        print("\n\n")
        mask_plot = [False if i == j else True for j in np.arange(len(all_data))]
        if verbose >= 1:
            print(f"Product {i} --> Mask: {mask_plot}")

        columns = all_data[i].columns[-(TOP_SIMILAR+1):].values
        if verbose >= 1:
            print(f"Products used in Model: {columns}")

        masked_complex_objective_plot = partial(complex_objective, ts_index_name=test_index_name, ts_index=test_index, mask=mask_plot, logT=LOG_TRANSFORM, verbose=0
                                               ,all_models=all_models,all_data=all_data)

        finalx = []
        finaly = []
        finalrev = []

        xs = allp[int(columns[0])]  # Main Product Price is in xs
        ys = allp[int(columns[1])]  # Exogenous Product Price in in ys

        if verbose >= 1:
            print(f"Price intervals used for X-axis (product {int(columns[0])}): {xs}")
            print(f"Price intervals used for Y-axis (product {int(columns[1])}): {ys}")
        
        for x, y in itertools.product(xs, ys):
            price_list = [0, 0, 0]

            # Fix price for product 0
            if int(columns[0]) == 0:  # If the main product is product 0
                price_list[0] = x
            elif int(columns[1]) == 0: # If exogenous product is product 0
                price_list[0] = y
            else:
                price_list[0] = 0

            # Fix price for product 1
            if int(columns[0]) == 1:  # If the main product is product 1
                price_list[1] = x
            elif int(columns[1]) == 1: # If exogenous product is product 1
                price_list[1] = y
            else:
                price_list[1] = 0

            # Fix price for product 2
            if int(columns[0]) == 2:  # If the main product is product 2
                price_list[2] = x
            elif int(columns[1]) == 2: # If exogenous product is product 2
                price_list[2] = y
            else:
                price_list[2] = 0

            rev = -masked_complex_objective_plot(price_list)
            finalx.append(x)
            finaly.append(y)
            finalrev.append(rev)   

        fig = surface3DChart(
            x=finalx, y=finaly, z=finalrev,
            title= 'Product ' + columns[0] + ' Revenue',
            xTitle= 'Product ' + columns[0] + ' Price',
            yTitle= 'Product ' + columns[1] + ' Price',
            width=1200,
            height=800            
            )

        filename = "".join(ChainMaster.split()) + "_" + "".join(ProdCat.split()) + "_Top" + str(TOP_PRODUCTS) + "_Sim" + str(TOP_SIMILAR) + \
            "_Log" + str(LOG_TRANSFORM) + "_Add" + str(ZERO_ADDER) + \
            "_Prod" + str(i) + "_Resample" + str(RESAMPLE_FREQ) + "_f" + str(FORECAST_PERIOD) + "_s" + str(SEASONAL_PERIOD)  + \
            "_Per" + str(period) + ".html"
        filenameFull = os.path.join(savePath,filename)
        if verbose >=1: print(filenameFull)
        filenames.append(filenameFull)
        py.plot(fig, filename = filenameFull,auto_open=displayPlots)
    return(filenames)
    

def runOptimizer(ProductsList,dataRaw,ChainMaster,modelsStats,leaveOutLastPeriods=0,verbose=0,ProdCat=''):
    opt_stats = pd.DataFrame()
    numProducts = len(ProductsList)
    opt_stats['Chain Master'] = [ChainMaster] * numProducts
    opt_stats['Product'] = ProductsList
    
    
    printLog("GET DATA",ChainMaster)
    all_data,all_data_non_transformed,colExog,colEnc,colDec = modelsLoadData(ProductsList,dataRaw,ChainMaster,leaveOutLastPeriods=leaveOutLastPeriods)
    
    printLog("TEST/TRAIN",ChainMaster)
    all_train, all_test, all_train_non_transformed, all_test_non_transformed = ModelsTestTrain(all_data,all_data_non_transformed)
    opt_stats['Actual Demand'] = [all_test_non_transformed[x]['9L Cases'].values[0] for x in np.arange(len(all_test_non_transformed))]
    opt_stats['Actual Price'] = [all_test_non_transformed[x].iloc[0][str(x)] for x in np.arange(len(all_test_non_transformed))]
    opt_stats['Actual Revenue'] =  [opt_stats['Actual Demand'][x] * opt_stats['Actual Price'][x]  for x in np.arange(numProducts)]
    opt_stats['Actual Chain Master Revenue'] =  [sum(opt_stats['Actual Revenue'])] *numProducts 
        
    printLog("NAIVE FORECAST",ChainMaster)
    all_models = modelsStats['Naive Best Model']
    naive_price, naive_demand, naive_revenue ,naive_total_revenue = opt_naive(all_models,all_test_non_transformed) #uses test price and predict demand based on naive
    opt_stats['Naive Prices'] = naive_price
    opt_stats['Naive Demand'] = naive_demand
    opt_stats['Naive Revenue'] = naive_revenue
    opt_stats['Naive Chain Master Revenue'] = [naive_total_revenue] * numProducts
    
    printLog("MASK",ChainMaster)
    mask = opt_get_mask(all_data,all_test)
    opt_stats['mask'] = mask
    
    printLog("SPACE",ChainMaster)
    space = opt_get_space(all_data)
    opt_stats['space'] = space
    
    printLog("Test Index",ChainMaster)
    test_index_name = 'WeekDate'
    test_index = all_test_non_transformed[0][test_index_name].values
    opt_stats['test_index'] = [test_index] * numProducts# for i in ProductsList]
    
    #############
    ## P0 Only ##
    if True:
        printLog("GET FUNCTION P0",ChainMaster)
        all_models = modelsStats['P0 Best Model']
        masked_complex_objective = opt_get_func(all_data=all_data
                                                ,all_models=all_models
                                                ,complex_objective=complex_objective
                                                ,test_index_name=test_index_name
                                                ,test_index=test_index
                                                ,mask=mask,verbose=verbose
                                                ,P0_only=True)
        opt_stats['masked_complex_objective'] = masked_complex_objective
        #### NON-OPTIMIZED ###
        printLog("GET NON-OPTIMIZED REVENUE P0",ChainMaster)
        out=opt_get_non_optimized(masked_complex_objective=masked_complex_objective
                                  ,all_test_non_transformed=all_test_non_transformed
                                  ,ChainMaster=ChainMaster)
        opt_stats = pd.concat([opt_stats,out],axis=1)
        
        #### OPTIMIZING ###
        printLog("OPTIMIZING P0",ChainMaster)
        out=opt_optimize(masked_complex_objective=masked_complex_objective
                         ,space=space
                         ,ChainMaster=ChainMaster)
        opt_stats = pd.concat([opt_stats,out],axis=1)

    ############
    ## P0+Sim ##
    if True:
        printLog("GET FUNCTION P0+Sim",ChainMaster)
        all_models = modelsStats['P0+Sim Best Model']
        masked_complex_objective = opt_get_func(all_data,all_models,complex_objective,test_index_name,test_index,mask,verbose=verbose,P0_only=False)
        opt_stats['masked_complex_objective'] = masked_complex_objective
        
        #### NON-OPTIMIZED ###
        printLog("GET NON-OPTIMIZED REVENUE P0+Sim",ChainMaster)
        out=opt_get_non_optimized(masked_complex_objective=masked_complex_objective
                                  ,all_test_non_transformed=all_test_non_transformed
                                  ,ChainMaster=ChainMaster)
        
        opt_stats = pd.concat([opt_stats,out],axis=1)
        
        #### OPTIMIZING ###
        printLog("OPTIMIZING P0+Sim",ChainMaster)
        out=opt_optimize(masked_complex_objective=masked_complex_objective
                         ,space=space
                         ,ChainMaster=ChainMaster)
        opt_stats = pd.concat([opt_stats,out],axis=1)
        
    ############
    # 3D Charts ##
    if  (TOP_SIMILAR==1):
        printLog("3D CHARTS",ChainMaster)
        filenames = opt_get_chart(all_data,all_models,space,ChainMaster,ProdCat,test_index,test_index_name,verbose=1,STEPS=5,displayPlots=False,period=leaveOutLastPeriods+1)
        #opt_stats['3d_chart_filenames']  = filenames
    
    printLog("COMPLETED",ChainMaster)
    
    return(opt_stats)

##################
#### FULL RUN ####
##################

def getFileName(name,ProdCat,extension):
    return(name +  "_"+ ProdCat + "_Prod" + str(TOP_PRODUCTS)+ "_Sim" + str(TOP_SIMILAR )+ "_Per" + ''.join(str(s) for s in RUN_PERIODS) + '.' + extension)
    
def saveDF(data,data_name,ProdCat):
    f = getFileName(data_name,ProdCat,'pkl')
    pd.to_pickle(data,f)
    print("Pickle File ",f," saved in filder: ",os.getcwd())
    
def fullRun(ProdCat,ChainMasters,dataRaw,savePKL = True, saveXLS = True):
    print("="*50)
    print("Parameters being used...")
    print("="*50)
    print(f"TOP_PRODUCTS = {TOP_PRODUCTS}")
    print(f"TOP_SIMILAR = {TOP_SIMILAR}")
    print(f"RUN_PERIODS = {RUN_PERIODS}")
    print("ProdCat:",ProdCat)
    print("ChainMasters:",ChainMasters)
    full_models=pd.DataFrame()
    full_opt=pd.DataFrame()
    
    for ChainMaster in ChainMasters:
        printLog("Products List ",[ProdCat,ChainMaster])
        ProductsList = getTopProducts(dataRaw, ChainMaster=ChainMaster, ProdCat=ProdCat, topN=TOP_PRODUCTS, timeCol='WeekDate')

        for period in RUN_PERIODS:
            printLog("AUTO-TS",[ProdCat,ChainMaster,str(period)])
            #### AUTO TS ######
            modelsStats=runModels(ProductsList=ProductsList
                                ,dataRaw=dataRaw
                                ,ChainMaster=ChainMaster
                                ,leaveOutLastPeriods=(period-1))
            modelsStats['Product Category']=ProdCat
            modelsStats['Period to Last']=period
            
            # append to generic data frame (fullstats)
            full_models=full_models.append(modelsStats,ignore_index=True)
            
            #### OPTIMIZER #####
            printLog("OPTIMIZER",[ProdCat,ChainMaster,str(period)])
            modelsStats =  modelsStats 
        
            printLog("Running Optimizer",[ProdCat,ChainMaster,str(period)])
            opt_stats=runOptimizer(ProductsList=ProductsList
                                   ,dataRaw=dataRaw
                                   ,ChainMaster=ChainMaster
                                   ,modelsStats=modelsStats
                                   ,verbose=0
                                   ,leaveOutLastPeriods=(period-1)
                                   ,ProdCat=ProdCat
                                   )
            
            opt_stats['Period to Last']=period
            # append to the generic data frame
            full_opt=full_opt.append(opt_stats,ignore_index=True)
            
    full_opt=full_opt.sort_values(['Chain Master','Product'])
    full_models=full_models.sort_values(['Chain Master','Product'])
    
    if savePKL:
        printLog("Saving to PICKLE",[ProdCat])
        saveDF(full_models[outputColsModels],"full_models",ProdCat)
        saveDF(full_opt[outputColsOptAll],"full_opt",ProdCat)
                
    full_summary = (full_opt.groupby(['Chain Master','Product'])
                .apply(lambda x: pd.Series({
                    'P0 Non-Opt Revenue':sum(x['P0 Non-Opt Revenue'])
                    ,'P0 Opt Revenue':sum(x['P0 Revenue'])
                    ,'P0 Opt Diff':sum(x['P0 Revenue']) - sum(x['P0 Non-Opt Revenue'])
                    ,'P0 Opt Diff Ratio':sum(x['P0 Revenue']) / sum(x['P0 Non-Opt Revenue'])
                    ,'P0+Sim Non-Opt Revenue':sum(x['P0+Sim Non-Opt Revenue'])
                    ,'P0+Sim Opt Revenue':sum(x['P0+Sim Revenue'])
                    ,'P0+Sim Opt Diff':sum(x['P0+Sim Revenue']) - sum(x['P0+Sim Non-Opt Revenue'])
                    ,'P0+Sim Opt Diff Ratio':sum(x['P0+Sim Revenue']) / sum(x['P0+Sim Non-Opt Revenue'])
                }))
                .reset_index()
               )
    full_chain_summary = (full_opt.groupby(['Chain Master'])
                .apply(lambda x: pd.Series({
                    'P0 Non-Opt Revenue':sum(x['P0 Non-Opt Revenue'])
                    ,'P0 Opt Revenue':sum(x['P0 Revenue'])
                    ,'P0 Opt Diff':sum(x['P0 Revenue']) - sum(x['P0 Non-Opt Revenue'])
                    ,'P0 Opt Diff Ratio':sum(x['P0 Revenue']) / sum(x['P0 Non-Opt Revenue'])
                    ,'P0+Sim Non-Opt Revenue':sum(x['P0+Sim Non-Opt Revenue'])
                    ,'P0+Sim Opt Revenue':sum(x['P0+Sim Revenue'])
                    ,'P0+Sim Opt Diff':sum(x['P0+Sim Revenue']) - sum(x['P0+Sim Non-Opt Revenue'])
                    ,'P0+Sim Opt Diff Ratio':sum(x['P0+Sim Revenue']) / sum(x['P0+Sim Non-Opt Revenue'])
                }))
                .reset_index()
               )
    
    
    if saveXLS:
        printLog("Saving to EXCEL",[ProdCat])
        f = getFileName('Output',ProdCat,'xlsx')
        writer = pd.ExcelWriter(f, engine='xlsxwriter') 
        #tot_revenue.to_excel(writer, sheet_name='Tot Revenue')
        full_chain_summary.to_excel(writer, sheet_name='Opt Chain Summary')
        full_summary.to_excel(writer, sheet_name='Opt Summary')
        full_opt[outputColsOptP0].to_excel(writer, sheet_name='Opt P0')
        full_opt[outputColsOptP0Sim].to_excel(writer, sheet_name='Opt P0_Sim')
        full_opt[outputColsOptNaive].to_excel(writer, sheet_name='Opt Naive')
        full_models[outputColsModels].to_excel(writer, sheet_name='Models')
        writer.save()
        print("EXCEL File ",f," saved in filder: ",os.getcwd())
    return(full_models,full_opt,full_summary,full_chain_summary)

