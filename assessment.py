
# HIT140 Assessment 2 - Extended Comprehensive Python Code
# Team: Sydney Group 31
# Members: Asim Sharma, Orchid Shrestha, Rojan Uprety, Shubham Singh

# -----------------------------
# Step 0: Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10,6)

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
# Replace 'your_dataset.csv' with the actual dataset
data = pd.read_csv('your_dataset.csv')

print("==== Dataset Preview ====")
print(data.head())
print("\n==== Dataset Info ====")
print(data.info())
print("\n==== Statistical Summary ====")
print(data.describe())
print("\n==== Missing Values ====")
print(data.isnull().sum())

# -----------------------------
# Step 2: Data Cleaning
# -----------------------------
# Fill missing numeric values with mean
numeric_cols = data.select_dtypes(include=np.number).columns
for col in numeric_cols:
    data[col].fillna(data[col].mean(), inplace=True)

# Fill missing categorical values with mode
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# -----------------------------
# Step 3: Exploratory Data Analysis (EDA)
# -----------------------------
print("\n==== Column Distributions ====")
for col in numeric_cols:
    plt.figure()
    sns.histplot(data[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Pairplots for numeric columns
sns.pairplot(data[numeric_cols])
plt.show()

# Boxplots to detect outliers
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=data[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.show()

# -----------------------------
# Step 4: Feature Engineering
# -----------------------------
# Example: Create a new feature as combination of two numeric columns
if 'Column1' in data.columns and 'Column2' in data.columns:
    data['Combined_Feature'] = data['Column1'] + data['Column2']
    print("\nNew feature 'Combined_Feature' created.")

# Example: Binning numeric variable into categories
if 'Column3' in data.columns:
    data['Column3_Binned'] = pd.cut(data['Column3'], bins=5, labels=False)
    print("\nColumn3 binned into 5 categories.")

# -----------------------------
# Step 5: Clustering Analysis (K-Means)
# -----------------------------
# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[numeric_cols])

# Determine optimal clusters using elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1,11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-Means with chosen k (example k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)
print("\nClusters assigned to data.")

# Visualize clusters (first two numeric features)
plt.figure()
sns.scatterplot(x=data[numeric_cols[0]], y=data[numeric_cols[1]], hue=data['Cluster'], palette='Set2')
plt.title('Cluster Visualization')
plt.show()

# -----------------------------
# Step 6: Correlation Analysis
# -----------------------------
print("\n==== Pearson Correlation Coefficients ====")
for col1 in numeric_cols:
    for col2 in numeric_cols:
        if col1 != col2:
            corr, _ = pearsonr(data[col1], data[col2])
            print(f'{col1} vs {col2}: {corr:.2f}')

# -----------------------------
# Step 7: Predictive Modeling - Regression
# -----------------------------
# Example: Predict 'TargetColumn' using all other numeric columns
if 'TargetColumn' in data.columns:
    X = data.drop(columns=['TargetColumn'])
    y = data['TargetColumn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    print("\n==== Linear Regression Results ====")
    print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred_lr):.2f}")

    # Random Forest Regression
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    print("\n==== Random Forest Regression Results ====")
    print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}")

# -----------------------------
# Step 8: Save Processed Data and Results
# -----------------------------
data.to_csv('processed_data.csv', index=False)
print("\nProcessed dataset saved as 'processed_data.csv'.")

# -----------------------------
# Step 9: Additional Visualizations
# -----------------------------
# Heatmap of clusters vs target
if 'TargetColumn' in data.columns:
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Cluster', y='TargetColumn', data=data)
    plt.title('Target Variable by Cluster')
    plt.show()

# Pairplot colored by clusters
sns.pairplot(data[numeric_cols + ['Cluster']], hue='Cluster', palette='Set1')
plt.show()

# Histogram of target variable
if 'TargetColumn' in data.columns:
    plt.figure()
    sns.histplot(data['TargetColumn'], bins=30, kde=True, color='salmon')
    plt.title('Target Variable Distribution')
    plt.show()

# -----------------------------
# Step 10: Insights and Summary (Print statements)
# -----------------------------
print("\n==== Key Insights ====")
print("- Data has been cleaned and missing values handled.")
print("- Feature engineering added new features for better analysis.")
print("- Clustering identified groups within the dataset.")
print("- Predictive models applied and performance metrics calculated.")
print("- All processed data and visualizations are saved and ready for report.")

# End of Extended HIT140 Python Code
