import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy import stats
from pandas import DataFrame as df


dict = df('d1': ['0','1','1','0','0','0','0'],
        'd2': ['1',1,0,1,1,0,1],
        'd3': [1,1,1,1,0,0,1],
        'd4': [0,1,1,0,0,1,1],
        'd5': [1,0,1,1,0,1,1],
        'd6': [0,0,1,1,1,1,1],
        'd7': [1,1,1,0,0,0,0],
        'd8': [1,1,1,1,1,1,1],
        'd9': [1,1,1,0,0,1,1],
        'd0': [1,1,1,1,1,1,0])



