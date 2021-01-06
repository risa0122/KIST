import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import pandas as pd
"""1. Collect the data"""
df = pd.read_table('wine.data',sep=',',names=['Alcohol','Malic_acid','Ash','Alcalinity of ash','Magnesium','Total phenols',
                                                 'Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue',
                                                 'OD280/OD315_of_diluted_wines','Proline'])
target = df.index
"""2. Normalize the data"""
df = StandardScaler().fit_transform(df)
"""3. Calculate the covariance matrix"""
COV = np.cov(df.T) # We have to transpose the data since the documentation of np.cov() sais
                   # Each row of `m` represents a variable, and each column a single
                   # observation of all those variables
"""4. Find the eigenvalues and eigenvectors of the covariance matrix"""
eigval,eigvec = np.linalg.eig(COV)
print(np.cumsum([i*(100/sum(eigval)) for i in eigval])) # As you can see, the first two principal components contain 55% of
                                                        # the total variation while the first 8 PC contain 90%
"""5. Use the principal components to transform the data - Reduce the dimensionality of the data"""
# The wine dataset is 13 dimensional and we want to reduce the dimensionality to 2 dimensions
# Therefore we use the two eigenvectors with the two largest eigenvalues and use this vectors
# to transform the original dataset.
# We want to have 2 Dimensions hence the resulting dataset should be a 178x2 matrix.
# The original dataset is a 178x13 matrix and hence the "principal component matrix" must be of
# shape 13*2 where the 2 columns contain the covariance eigenvectors with the two largest eigenvalues
PC = eigvec.T[0:2]
data_transformed = np.dot(df,PC.T) # We have to transpose PC because it is of the format 2x178
# Plot the data
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.scatter(data_transformed.T[0],data_transformed.T[1])
for l,c in zip((np.unique(target)),['red','green','blue']):
    ax0.scatter(data_transformed.T[0,target==l],data_transformed.T[1,target==l],c=c,label=l)
ax0.legend()
plt.show()