import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


# Function compute Entropy
def H(p):
    id_p = np.where(p != 0)
    return -np.sum(p[id_p] * np.log(p[id_p]))


# Initialize
p = np.array([0.1, 0.2, 0.3, 0.4, 0])

# Compute H(X)
print("H = ", H(p))     # H = 1.27985422583

##################

# Initialize
theta = np.arange(0, 1, 0.01)
p = np.array([1-theta, theta/4, theta/4, theta/4, theta/4]).T
# p 의 한 행의 합은 1

# Visualize
HX = []         # 엔트로피 계산 결과, 행벡터로 만들기
id_max = 0
H_max = 0
for i in range(len(theta)):
    temp = H(p[i])
    HX.append(temp)
    if (temp > H_max):
        H_max = temp
        id_max = i

plt.plot(p[:,1], HX)
plt.plot(p[id_max,1], H_max, 'ro')
plt.show()

