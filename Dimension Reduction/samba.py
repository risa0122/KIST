import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sympy import Symbol, solve
from skimage import io
import matplotlib.pyplot as plt

np.random.seed(0)


x1 = np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})

x1 = np.random.rand(9).reshape((3,3)) # 0~9까지의 3*3 배열 생성

#print(x1)
#print(x1.shape)

x2 = np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})

#x2 = x1.T

#### manual inverse matrix ####
x2_1 = x1[0:,0]
x2_2 = x1[0:,1]
x2_3 = x1[0:,2]
#print(x2_1)
#print(x2_1.shape)
x2 = np.stack((x2_1,x2_2,x2_3),axis=0)

#print(x2.shape)

x3 = np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})
x3 = np.copy(x2)
x3[0,:] = x2[0,:]*1.2
x3[1,:] =x2[1,:]*1.3
x3[2,:] = x2[2,:]*0.4


######### 정규화 #########

StandardScaler(x1,x2,x3)



########### 1열 & 1변수1원소  전환  ##############

x1 = np.array(x1,ndmin=3)  #(3,3,1): 3행1열이 3번반복.> 3차원으로 만들어
#print(x1)
x1 = x1.T.reshape(9,1)      # 9 개의 벡터를 가진 행렬로 만들기
x2 = np.array(x2,ndmin=3)
x2 = x2.T.reshape(9,1)
x3 = np.array(x3,ndmin=3)
x3 = x3.T.reshape(9,1)
#print(x1.shape)
#print(x1)

x = np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})

x = np.stack((x1,x2,x3),axis=-1)
#print(x.shape)
x = x.reshape(9,3)
#print(x)


'''
x1 = np.transpose(x1)
x = np.dot(x1,x2)
#print(x)
#print(x.shape)

########## Calculate the covariance matrix ##########

avg = sum(x,0.0)/len(x)   # 0.30


o = np.zeros((9,1))

for i in x:
    for a in range(9):
       y = (i - avg) / len(x)
       o[a] = y * y.T


y = []
for i in x:
    z = (i - avg)
    zz = np.mean(z * z.T)09
    y.append(zz)

y = np.matrix(y).reshape((9,1))
#print(y)

print(np.cov(x1.all(),x2.all(),x3.all()))
'''
#print(x[1])

xx = []
for i in range(9):
    for j in range(9):
        yy = np.dot(x[i],x[j])
        xx.append(yy)
xx = np.matrix(xx).reshape(9,9)
#print(xx)

######### 고유값 , 고유벡터 #########
# det(A−λI)=0   # I: 단위행렬

'''
A = xx
lam = Symbol('lam')
lamI = np.eye(9)*lam   # 9*9 단위행렬 생성
#print(lamI.shape)
#print(A - lamI)

aa = A - lamI
aa = aa.reshape((9,1,9))

a1 = aa[0]
a2 = aa[0]


b = np.zeros((1,1))
det = np.linalg.solve(aa, b)    # 행렬식 계산에러 > 미지수 포함 행렬식 계산 불가
print(det)
#solve(det,lam)
'''

eig_val, eig_vec = np.linalg.eig(xx)
print(eig_vec)

##### 데이터 차원 축소 #####

aa = eig_vec[0]
aaa = eig_vec[1]
aaaa = eig_vec[2]
aaaaa = np.stack((aa,aaa,aaaa),axis=0)
print(aaaa.shape)

sc = x * aaaaa

io.imshow(sc)   # in Jupyter