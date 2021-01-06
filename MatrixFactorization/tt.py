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

# _____________    _______________


new_A = 0.45
new_B = 0.75

for it in range(350):
    def Total(new_A,new_B,r=0.01):

        YP = new_A + (new_B * X)
        SSE_A = -(Y-YP)
        SSE_B = (-(Y-YP))*X

        new_A = new_A - (r * (SSE_A))
        new_B = new_B - (r * (SSE_B))

        SSEf = []
        for i in range(10):
            SSEf.append(0.5 * (Y[i] - YP[i]) ** 2)
            summ = np.sum(SSEf)
        return new_A, new_B,SSE_A,SSE_B
    print('Total SSE :',np.sum(Total(summ)))

Total(new_A,new_B)





