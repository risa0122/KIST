# https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/'

# L,U 사용 목적 :
# LU 분해는 선형방정식 Ax = B 를 풀 때 사용
# L(Ux)=b  -> Ux = L-1B 와 원래 Ax = B 비교,
# 1) 계산 편리함 computational convenience
# > 더 많은 계산(덧샘,곱샘) but 부동소수점같은 부정확&복잡 연산 적은 자릿수 데이터 계산
# 2) 분석 용이성 analytic simplicity
# x 찾는 문제, Ux = y, Ly = b 이용해서 x 찾음
#   y = b/L    ,    x = b/LU

## 치환행렬 permutation matrix : change the row sequence
import numpy as np
from scipy.linalg import lu

A = np.array(([1,1,1,1],
             [2,3,4,3],
             [3,4,6,9],
              [4,5,3,6]))

B = np.array(([6],[20],[29],[30]))
#A = np.loadtxt('/Users/taco-lab/PycharmProjects/Matix/make_numpy')

A = np.array(A)
np.set_printoptions(formatter={'float_kind':lambda x: '{0:0.2f}'.format(x)})
P,L,U = lu(A)
print('L',L)
print("U",U)

B = np.dot(P,L)
print(B.shape)  # 7*7

C = np.dot(B,U)
print(C.shape)  # 7*81

if A.all() == C.all():
    print('A = LU')
else:
    print('A =! LU')



