# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and Explore the Dataset
# Load the dataset
try:
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                     header=None, 
                     names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
except Exception as e:
    print("Error loading the dataset:", e)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nDataset info:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean the dataset (if needed)
# In this case, there are no missing values, but we could fill or drop them if there were.
# df.dropna(inplace=True)  # Example of dropping missing values

# Step 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Perform groupings and compute mean
average_by_class = df.groupby('class').mean()
print("\nAverage values by class:")
print(average_by_class)

# Step 3: Data Visualization
# Set the style for seaborn
sns.set(style="whitegrid")

# 1. Line chart (not applicable for this dataset, so we will skip it)

# 2. Bar chart: Average petal length per species
plt.figure(figsize=(10, 5))
average_by_class['petal_length'].plot(kind='bar', color='skyblue')
plt.title('Average Petal Length by Iris Species')
plt.xlabel('Iris Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('average_petal_length.png')
plt.show()

# 3. Histogram: Distribution of sepal length
plt.figure(figsize=(10, 5))
sns.histplot(df['sepal_length'], bins=10, kde=True, color='orange')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('sepal_length_distribution.png')
plt.show()

# 4. Scatter plot: Sepal length vs. Petal length
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='class', style='class', s=100)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Iris Species')
plt.tight_layout()
plt.savefig('sepal_vs_petal_length.png')
plt.show()

# Findings and Observations
print("\nFindings:")
print("1. The average petal length varies significantly among different species.")
print("2. The distribution of sepal length shows a normal distribution.")
print("3. There is a clear separation in petal length based on species, indicating distinct classes.")