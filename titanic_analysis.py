import pandas as pd
import seaborn as sns
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


plt.figure(figsize = (10, 5)) 

plt.subplot(1, 2, 1)  # First subplot
sns.histplot(df["age"], kde=True)
plt.title('Age Distribution')

plt.subplot(1, 2, 2)  # Second subplot
sns.histplot(df["fare"], kde=True)
plt.title('Fare Distribution')


removed_df = df[df["age"]<62]
sns.boxplot(removed_df["age"])
plt.tight_layout()  
plt.show()
