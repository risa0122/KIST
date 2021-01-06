from sympy import Symbol,solve,Function
import numpy as np

z = Symbol('z')

x = np.array([[z,2],[0,0]])
y = np.array([[2,0],[1,3]])
print(x-y)
a = x-y
np.linalg.det(a)


