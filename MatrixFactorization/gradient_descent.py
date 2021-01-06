# 'https://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html

import pandas as pd
from pandas import DataFrame as df
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x: '{0:0.2f}'.format(x)})

a = [1400,1600,1700,1875,1100,1550,2350,2450,1425,1700]
b = [245000,312000,279000,308000,199000,219000,405000,324000,319000,255000]
b = np.array(b)/1000

c =[]
for i in range(10):
   c.append((a[i],b[i]))
c.sort()


X = []
Y = []
for i in range(10):
    X.append((c[i][0] - min(a)) / (max(a) - min(a)))

for i in range(10):
    Y.append((c[i][1]-min(b))/(max(b)-min(b)))

X = np.round(X,2)
Y = np.round(Y,2)



A = 0.45
B = 0.75

#### Step 1 ####
# a,b 랜덤변수로 두고 에러 예측값(SSE) 구함


#Sum of Squared Errors

def SSE(Y,YP):
    SSEf = []
    YP = A + (B * X)
    for i in range(10):
        SSEf.append(0.5 * (Y[i] - YP[i]) ** 2)
    print('Total SSE :', np.sum(SSEf))

# Goal :  finds the optimal weights (A,B)
# that reduces  prediction error.


### Step 2 ###
# w,r,t

SSA = []
SSB = []

def SSE_A(Y,YP):
    SSA.append(-(Y-YP))

def SSE_B(Y,YP):
    SSB.append((-(Y-YP))*X)

print('∂SSE/∂A :', np.sum(SSE_A(Y,YP)))
print('∂SSE/∂B :',np.sum(SSE_B(Y,YP)))


### Step 3 ###
      #running rate
newA = []
newB = []
def new_A(r,Y,YP):
    newA.append(newA-(r * (SSE_A(Y, YP))))
def new_B(r,Y,YP):
    newB.append(newB-(r * (SSE_B(Y, YP))))

A = A
B = B
YP = YP
for i in range(5):
    r = 0.01
    new_A = A - (r*(SSE_A(Y,YP)))
    A = newA
    YP = new_YP(Y,YP)
    new_B = B - (r*(SSE_B(Y,YP)))
    B = newB
    new_YP = new_A+((new_B)*X)

    SSEf(Y,new_YP)














