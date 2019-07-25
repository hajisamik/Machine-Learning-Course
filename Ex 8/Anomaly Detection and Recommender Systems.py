# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:27:01 2018

@author: Abolfazl Hajisami
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


data = loadmat('data/ex8data1.mat')
X = data['X']


fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])



def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    
    return mu, sigma

mu, sigma = estimate_gaussian(X)


Xval = data['Xval']
yval = data['yval']