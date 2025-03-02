import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")
df.head()
df.describe()
print(df.size)
print(df.shape)
print(df["age"].mean())
print(df["age"].median())
print(df["age"].mode())
print(df["age"].std())
print(df["age"].var())
print(df.columns)
print(df["fare"].skew())
df["fare"].hist()
plt.show()  

