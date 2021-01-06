import pandas as pd
import surprise
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

data = surprise.Dataset.load_builtin('ml-100k')
df = pd.DataFrame(data.raw_ratings, columns=['user','item','rate','id'])
del df['id']
print(df)

df_table = df.set_index(['user','item']).unstack()
print(df_table.shape)

print(df_table.iloc[212:222, 808:817].fillna(""))

plt.imshow(df_table)
plt.grid(False)
plt.xlabel('item')
plt.ylabel('user')
plt.title('Rate Matrix')
plt.show()

algo = surprise.NMF(n_factors = 100)
cross_validate(algo,data)['test_mae'].mean()

