# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 22:22:16 2020

@author: annam
"""

import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import numpy as np


def surface3DChart(x,y,z,width=800,height=600
                   ,title=''
                   ,xTitle = 'Price Product 1', yTitle = 'Price Product 2', zTitle='Revenue'
                   ,drawMax = True):
    from scipy.interpolate import griddata
    #based on https://stackoverflow.com/questions/36589521/how-to-surface-plot-3d-plot-from-dataframe
    
    # 2D-arrays from inputs
    x1 = np.linspace(min(x), max(x), len(np.unique(x)))
    y1 = np.linspace(min(y), max(y), len(np.unique(y)))
    
    """
    x, y via meshgrid for vectorized evaluation of
    2 scalar/vector fields over 2-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.
    """
    x2, y2 = np.meshgrid(x1, y1)
    
    # Interpolate unstructured D-dimensional data.
    z2 = griddata((x,y), z, (x2, y2), method='cubic')
        
    fig =  go.Figure(data=[go.Surface(z=z2, x=x2, y=y2)])
    if(drawMax):
        zMax = max(z)
        zMaxLS =  np.linspace(zMax,zMax,1)
        xMax = np.array(x)[z==zMax]
        yMax = np.array(y)[z==zMax]
        fig.add_trace(go.Scatter3d(x=xMax,y=yMax,z=zMaxLS,mode='markers'))
        
    fig.update_layout(title=title, autosize=True
                  ,width=width, height=height
                  ,margin=dict(l=65, r=50, b=65, t=90)
                  ,scene = dict(
                      xaxis_title = xTitle
                      ,yaxis_title = yTitle
                      ,zaxis_title = zTitle
                  )
                  ,legend_title_text = zTitle
                  )
#    plot(fig)
    #fig.show()
    
    
    return(fig)
    
        
def testChart3D():
    x = [100,100,100,100,100,110,110,110,110,110,120,120,120,120,120]
    y = [10,11,12,13,14,10,11,12,13,14,10,11,12,13,14]
    z= np.random.rand(len(x)) * 100000  #Revennue
    plot(surface3DChart(x,y,z,title='TESTING'))
        