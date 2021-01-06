import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
from skimage.transform import rotate
from sklearn.decomposition import NMF
from sklearn.utils.validation import check_is_fitted


x = np.array([[10, 100,30],
               [70, 50, 70],
               [30, 100, 10]])
x = x/255
x1 = rotate(x,90)
x2 = rotate(x1,90)

y = rotate(x,270)
y1 = rotate(x1,270)
y2 = rotate(x2,270)

z = np.sort(x,axis=1)[::-1]
z1 = rotate(z,90)
z2 = rotate(z,180)

xx = np.hstack((x,x1,x2))
yy = np.hstack((y,y1,y2))
zz = np.hstack((z,z1,z2))
t1 = np.vstack((xx,yy,zz))   # 9*9

t2 = np.copy(t1)
t2[t2<=30] = t2[t2<=30]*1.5

t3 = np.copy(t1)
t3[t3<=50] = t3[t3<=50]*2

t4 = np.copy(t1)
t4[t4<=70] = t4[t4<=70]*2.5

t5 = np.copy(t1)
t5[t5==30] = t5[t5==30]*3

t6 = np.copy(t1)
t6[t6==50] = t6[t6==50]*3.5

t7 = np.copy(t1)
t7[t7==70] = t7[t7==70]*4

tt1 = t1.reshape((1,81))
tt2 = t2.reshape((1,81))
tt3 = t3.reshape((1,81))
tt4 = t4.reshape((1,81))
tt5 = t5.reshape((1,81))
tt6 = t6.reshape((1,81))
tt7 = t7.reshape((1,81))

tt = np.vstack((tt1,tt2,tt3,tt4,tt5,tt6,tt7))
X = tt

np.savetxt('make_numpy',X)

'''
plt.figure(figsize=(8, 7))

plt.subplot(1, 2, 1)
#plt.imshow(t1, cmap=plt.cm.get_cmap('RdBu', 9))
plt.imshow(t1)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
#plt.imshow(t4, cmap=plt.cm.get_cmap('RdBu', 9))
plt.imshow(t4)
plt.title('90')
plt.axis('off')
'''

# MF
from EASY_MF import MatrixFactorization

# P, Q is (81 X k), (k X 7) matrix
factorizer = MatrixFactorization(X, k=2, learning_rate=0.01, reg_param=0.01, epochs=300, verbose=True)
factorizer.fit()
factorizer.print_results()


# NMF

model = NMF(n_components=2, init='random', random_state=0,beta_loss='kullback-leibler',solver='mu')
W = model.fit_transform(X)
hh = model.components_
hh = hh.T

# entropy
def H(p):
    id_p = np.where(p != 0)
    return -np.sum(p[id_p] * np.log(p[id_p]))


print("entropy H(X) = ", H(tt))



def cross_entropy(p, q):
	return -sum([p[i]*np.log(q[i]) for i in range(len(p))])
Y =np.matmul(W,hh.T)

xx = cross_entropy(tt,Y)
print('cross entropy H(X,Y):',xx)

# KL divergence
def KL(p,q):
    return H(p) - cross_entropy(p,q)
aa = KL(X,Y)
print('KL divergence H(X)-H(X,Y):',aa)



# Mutual Info
def MI(p,q):
    return H(p)+H(q)-cross_entropy(p,q)
bb = MI(X,Y)
print('Mutual Information H(X)+H(Y)-H(X,Y):', bb)


'''#fail
import seaborn as sns
XX = sns.distplot(X, rug=True)
YY = sns.distplot(Y, rug=True)

plt.title('KL(P||Q) = %1.3f' % KL(XX, YY))
plt.plot(x, XX)
plt.plot(x, YY, c='red')
plt.show()
'''

A = []
B = []
for i in range(7):
    A.append(H(X[i]))

for i in range(7):
    B.append(cross_entropy(X[i],Y[i]))

A = np.array(A)
B = np.array(B)

'''
plt.title('KL(P||Q) = %0.22f' % KL(A, B))
plt.plot(np.arange(7),B, c='red')
plt.plot(np.arange(7),A)
plt.show()

plt.plot(,A,,B,'r-')
'''
# mi
C = []
D = []
for i in range(7):
    C.append(H(X[i]))
for i in range(7):
    D.append(H(Y[i])-cross_entropy(X[i],Y[i]))

C = np.array(C)
D = np.array(D)

plt.title('MI(P||Q) = %0.22f' % MI(X,Y))
plt.plot(x, C)
plt.plot(x, D, c='red')
plt.show()


#
def cond_entro(v):
    H(X)sum(H(X),H(Y))

