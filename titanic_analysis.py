import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder  # Corrected import

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")

# Display basic information about the dataset
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

# Create subplots for Age and Fare distributions
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)  # First subplot
sns.histplot(df["age"], kde=True)
plt.title('Age Distribution')

plt.subplot(1, 2, 2)  # Second subplot
sns.histplot(df["fare"], kde=True)
plt.title('Fare Distribution')

# Filter out rows where "age" is less than 62 and plot a boxplot
removed_df = df[df["age"] < 62]
sns.boxplot(x=removed_df["age"])

# Handle missing values
print(df.isnull().sum())  # Print the number of missing values for each column

# Fill missing "age" values with the median
df["age"].fillna(df["age"].median(), inplace=True)

# Fill missing "embarked" and "embark_town" values with the mode
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])  # Fixed chained assignment
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])  # Fixed chained assignment

# Handle missing "deck" values by adding a new category "H" and filling missing values with "H"
df["deck"] = df["deck"].fillna("H")  # Fixed inplace warning

# Check for missing values again
print(df.isnull().sum())  # This should now show no missing values for "age", "embarked", "embark_town", or "deck"

# Check for duplicates in the dataset
print(df.duplicated().sum())

# Define categorical columns (assuming these columns are categorical)
cat_cols = ["embarked", "embark_town", "deck", "sex"]  

# Label encoding for categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Define the feature columns (exclude "survived" column)
feature_cols = df.drop(columns=["survived"]).columns  # Fixed column drop

# Scaling the features (standard scaling and min-max scaling)
scaler = StandardScaler()

# Only apply scaling to numeric columns
numeric_cols = df[feature_cols].select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Display the plots
plt.tight_layout()  
plt.show()

