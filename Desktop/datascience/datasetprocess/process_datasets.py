import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Open file dialog to select datasets
Tk().withdraw()  # hides the main tkinter window
file1 = askopenfilename(title="Select dataset1.csv")
file2 = askopenfilename(title="Select dataset2.csv")

# Load CSV files
dataset1 = pd.read_csv(file1)
dataset2 = pd.read_csv(file2)


# Preview first few rows 
print("Dataset1 Head:")
print(dataset1.head())
print("\nDataset2 Head:")
print(dataset2.head())

# Check for missing values
print("\nMissing Values in Dataset1:")
print(dataset1.isnull().sum())
print("\nMissing Values in Dataset2:")
print(dataset2.isnull().sum())

# Convert time columns to datetime
dataset1['start_time'] = pd.to_datetime(dataset1['start_time'])
dataset1['rat_period_start'] = pd.to_datetime(dataset1['rat_period_start'])
dataset1['rat_period_end'] = pd.to_datetime(dataset1['rat_period_end'])
dataset2['time'] = pd.to_datetime(dataset2['time'])

# Example: Create new features
dataset1['seconds_since_rat_arrival'] = dataset1['seconds_after_rat_arrival']  # already exists
dataset1['risk_behavior'] = dataset1['risk']  # 0 = avoidance, 1 = risk-taking
