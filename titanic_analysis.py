import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder  # Corrected import
from sklearn.model_selection import train_test_split  # Ensure proper import

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")

# Display basic information about the dataset
print(df.head())  # Display the first few rows
print(df.describe())  # Summary statistics for numerical columns
print(f"Size: {df.size}")
print(f"Shape: {df.shape}")
print(f"Mean of age: {df['age'].mean()}")
print(f"Median of age: {df['age'].median()}")
print(f"Mode of age: {df['age'].mode()}")
print(f"Standard deviation of age: {df['age'].std()}")
print(f"Variance of age: {df['age'].var()}")
print(f"Columns: {df.columns}")
print(f"Skew of fare: {df['fare'].skew()}")

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
print(f"Duplicate rows: {df.duplicated().sum()}")

# Define categorical columns (assuming these columns are categorical)
cat_cols = ["embarked", "embark_town", "deck", "sex"]

# Label encoding for categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Define the feature columns (exclude "survived" column)
feature_cols = df.drop(columns=["survived", "Unnamed: 0"]).columns  # Excluding 'Unnamed: 0' (index column)

# Scaling the features (standard scaling and min-max scaling)
scaler = StandardScaler()

# Only apply scaling to numeric columns
numeric_cols = df[feature_cols].select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Correlation Matrix (Now, it will work since only numeric columns are included)
correlation_matrix = df[numeric_cols].corr()  # Only pass numeric columns for correlation

# Generate a heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')

# Split the dataset into features (X) and target (y)
X = df.drop(columns=["survived"])  # Features (independent variables)
y = df["survived"]  # Target variable (dependent variable)

# Split the data into training and testing sets (33% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# Display the plots
plt.tight_layout()  
plt.show()
