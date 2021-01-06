import numpy as np

def cross_entropy(predictions, targets, epsilon=1e-12):

    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

predictions = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.96]])
targets = np.array([[0,0,0,1],
                   [0,0,0,1]])
ans = 0.71355817782  #Correct answer
x = cross_entropy(predictions, targets)
print(np.isclose(x,ans))
