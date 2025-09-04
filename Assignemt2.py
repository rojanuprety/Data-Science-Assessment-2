import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os

# This will print current working directory
print("Current Working Directory:", os.getcwd())

# It will load datasets with error handling
try:
    df1 = pd.read_csv("dataset1.csv")
    df2 = pd.read_csv("dataset2.csv")
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure 'dataset1.csv' and 'dataset2.csv' are in the same directory as this script.")
    exit()

# --- DESCRIPTIVE ANALYSIS ---

# 1. Basic overview
print("Dataset1 shape:", df1.shape)
print("Dataset2 shape:", df2.shape)
print(df1.head())
print(df2.head())

# 2. Distribution of risk-taking and reward
risk_counts = df1['risk'].value_counts(normalize=True)
reward_counts = df1['reward'].value_counts(normalize=True)
print("Risk-taking proportion:\n", risk_counts)
print("Reward proportion:\n", reward_counts)

# This code will plot risk-taking vs avoidance distribution
sns.countplot(x='risk', data=df1, palette="Set2")
plt.title("Risk-taking vs Avoidance")
plt.xlabel("Risk (0=Avoid, 1=Risk-taking)")
plt.ylabel("Count")
plt.show()

# This will plot reward distribution chart
sns.countplot(x='reward', data=df1, palette="Set1")
plt.title("Reward vs No Reward")
plt.xlabel("Reward (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

# --- RELATIONSHIP BETWEEN RISK & REWARD ---

# Cross-tabulation
ct = pd.crosstab(df1['risk'], df1['reward'])
print("Risk vs Reward Table:\n", ct)

# Chi-square test
chi2, p, dof, expected = chi2_contingency(ct)
print(f"Chi-square: {chi2:.3f}, p-value: {p:.4f}")

# Stacked bar plot
(ct.div(ct.sum(1), axis=0)).plot(kind="bar", stacked=True)
plt.title("Proportion of Rewards within Risk Strategies")
plt.xlabel("Risk (0=Avoid, 1=Risk-taking)")
plt.ylabel("Proportion")
plt.show()

# --- SEASONAL ANALYSIS ---

# Average risk-taking by season
season_risk = df1.groupby('season')['risk'].mean()
print("Average risk-taking by season:\n", season_risk)

season_risk.plot(kind='bar', color='orange')
plt.title("Average Risk-taking by Season")
plt.xlabel("Season (0=Winter, 1=Spring)")
plt.ylabel("Proportion Risk-taking")
plt.show()

# --- USING DATASET 2 (contextual checks) ---

# Comparingrat arrivals vs bat landings
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', data=df2)
plt.title("Rat Arrivals vs Bat Landings")
plt.xlabel("Rat Arrival Number")
plt.ylabel("Bat Landing Number")
plt.show()

# Correlation check
print("Correlation matrix:\n", df2[['rat_arrival_number', 'bat_landing_number', 'food_availability']].corr())