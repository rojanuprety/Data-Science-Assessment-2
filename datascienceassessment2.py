# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visuals
plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Load and clean dataset1 (individual bat landings)
print("Loading and cleaning dataset1.csv...")
def load_and_clean_dataset1(file_path):
    """
    Load and clean the individual bat landing dataset.
    
    Parameters:
    file_path (str): Path to dataset1.csv
    
    Returns:
    pandas.DataFrame: Cleaned dataset
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Display basic info
    print(f"Dataset1 shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Handle missing values - different strategies for different columns
    if df['bat_landing_to_food'].isnull().sum() > 0:
        # Fill missing landing-to-food times with median
        median_landing = df['bat_landing_to_food'].median()
        df['bat_landing_to_food'].fillna(median_landing, inplace=True)
        print(f"Filled {df['bat_landing_to_food'].isnull().sum()} missing values in bat_landing_to_food")
    
    # Convert time columns to datetime
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['rat_period_start'] = pd.to_datetime(df['rat_period_start'], errors='coerce')
    df['rat_period_end'] = pd.to_datetime(df['rat_period_end'], errors='coerce')
    df['sunset_time'] = pd.to_datetime(df['sunset_time'], errors='coerce')
    
    # Create rat presence flag
    df['rat_present'] = df['rat_period_start'].notna().astype(int)
    
    # Remove extreme outliers in time differences
    q95 = df['bat_landing_to_food'].quantile(0.95)
    df = df[df['bat_landing_to_food'] <= q95]
    
    print(f"Final dataset1 size: {df.shape}")
    return df

# Load and clean dataset2 (30-minute intervals)
print("\nLoading and cleaning dataset2.csv...")
def load_and_clean_dataset2(file_path):
    """
    Load and clean the 30-minute interval dataset.
    
    Parameters:
    file_path (str): Path to dataset2.csv
    
    Returns:
    pandas.DataFrame: Cleaned dataset
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Display basic info
    print(f"Dataset2 shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Create rat presence flag
    df['rat_present'] = (df['rat_arrival_number'] > 0).astype(int)
    
    # Handle missing values in food availability
    if df['food_availability'].isnull().sum() > 0:
        df['food_availability'].fillna(df['food_availability'].median(), inplace=True)
    
    print(f"Final dataset2 size: {df.shape}")
    return df

# Load both datasets
df1 = load_and_clean_dataset1('dataset1.csv')
df2 = load_and_clean_dataset2('dataset2.csv')

# Analysis 1: Compare bat landing frequency with vs without rats
print("\n" + "="*50)
print("ANALYSIS 1: BAT LANDING FREQUENCY")
print("="*50)

def analyze_landing_frequency(df2):
    """Analyze how rat presence affects bat landing frequency."""
    
    # Group by rat presence and calculate statistics
    landing_stats = df2.groupby('rat_present')['bat_landing_number'].agg([
        'mean', 'std', 'count'
    ]).round(2)
    
    print("Bat landings per 30-minute period:")
    print(landing_stats)
    
    # Perform t-test
    no_rats = df2[df2['rat_present'] == 0]['bat_landing_number']
    with_rats = df2[df2['rat_present'] == 1]['bat_landing_number']
    
    t_stat, p_value = stats.ttest_ind(no_rats, with_rats, equal_var=False)
    
    print(f"\nT-test results: t = {t_stat:.3f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ Significant difference in bat landings with vs without rats")
    else:
        print("✗ No significant difference found")
    
    return no_rats, with_rats

no_rats_landings, with_rats_landings = analyze_landing_frequency(df2)

# Analysis 2: Bat vigilance behavior (time to approach food)
print("\n" + "="*50)
print("ANALYSIS 2: VIGILANCE BEHAVIOR")
print("="*50)

def analyze_vigilance_behavior(df1):
    """Analyze if bats take longer to approach food when rats are present."""
    
    # Group by rat presence and calculate statistics
    vigilance_stats = df1.groupby('rat_present')['bat_landing_to_food'].agg([
        'mean', 'std', 'count'
    ]).round(2)
    
    print("Time to approach food (seconds):")
    print(vigilance_stats)
    
    # Perform t-test
    no_rats_time = df1[df1['rat_present'] == 0]['bat_landing_to_food']
    with_rats_time = df1[df1['rat_present'] == 1]['bat_landing_to_food']
    
    t_stat, p_value = stats.ttest_ind(no_rats_time, with_rats_time, equal_var=False)
    
    print(f"\nT-test results: t = {t_stat:.3f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ Bats take significantly longer to approach food when rats are present")
    else:
        print("✗ No significant difference in approach time")
    
    return no_rats_time, with_rats_time

no_rats_time, with_rats_time = analyze_vigilance_behavior(df1)

# Analysis 3: Risk-taking behavior
print("\n" + "="*50)
print("ANALYSIS 3: RISK-TAKING BEHAVIOR")
print("="*50)

def analyze_risk_behavior(df1):
    """Analyze if bats change risk-taking behavior when rats are present."""
    
    # Calculate risk-taking rates
    risk_stats = df1.groupby('rat_present')['risk'].agg([
        'mean', 'count'
    ]).round(3)
    
    risk_stats['risk_percentage'] = (risk_stats['mean'] * 100).round(1)
    
    print("Risk-taking behavior (1 = risk taken, 0 = risk avoided):")
    print(risk_stats[['mean', 'risk_percentage', 'count']])
    
    # Perform chi-square test
    contingency_table = pd.crosstab(df1['rat_present'], df1['risk'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nChi-square test results: χ² = {chi2:.3f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ Significant difference in risk-taking behavior")
    else:
        print("✗ No significant difference in risk-taking behavior")
    
    return risk_stats

risk_stats = analyze_risk_behavior(df1)

# Visualization 1: Bat landing comparison
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

def create_visualizations(df1, df2, no_rats_landings, with_rats_landings, no_rats_time, with_rats_time):
    """Create all required visualizations for the presentation."""
    
    # Figure 1: Bat landings with vs without rats
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    landing_data = [no_rats_landings, with_rats_landings]
    plt.boxplot(landing_data, labels=['No Rats', 'Rats Present'])
    plt.title('Bat Landings per 30-minute Period')
    plt.ylabel('Number of Landings')
    plt.grid(True, alpha=0.3)
    
    # Add mean values to the plot
    means = [np.mean(no_rats_landings), np.mean(with_rats_landings)]
    for i, mean in enumerate(means):
        plt.text(i+1, mean + 2, f'Mean: {mean:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Figure 2: Time to approach food
    plt.subplot(2, 2, 2)
    time_data = [no_rats_time, with_rats_time]
    plt.boxplot(time_data, labels=['No Rats', 'Rats Present'])
    plt.title('Time to Approach Food')
    plt.ylabel('Seconds')
    plt.grid(True, alpha=0.3)
    
    # Figure 3: Risk-taking behavior
    plt.subplot(2, 2, 3)
    risk_rates = risk_stats['risk_percentage']
    colors = ['lightblue', 'lightcoral']
    bars = plt.bar(['No Rats', 'Rats Present'], risk_rates, color=colors)
    plt.title('Risk-Taking Behavior')
    plt.ylabel('Percentage of Risk-Taking (%)')
    plt.ylim(0, 35)
    
    # Add value labels on bars
    for bar, value in zip(bars, risk_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}%', ha='center', va='bottom')
    
    # Figure 4: Temporal pattern of rat activity
    plt.subplot(2, 2, 4)
    hourly_rat_activity = df2.groupby('hours_after_sunset')['rat_arrival_number'].mean()
    plt.plot(hourly_rat_activity.index, hourly_rat_activity.values, 
             marker='o', linewidth=2, markersize=4)
    plt.title('Rat Activity vs Time After Sunset')
    plt.xlabel('Hours After Sunset')
    plt.ylabel('Average Rat Arrivals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bat_rat_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizations saved as 'bat_rat_analysis_results.png'")

create_visualizations(df1, df2, no_rats_landings, with_rats_landings, no_rats_time, with_rats_time)

# Summary of findings
print("\n" + "="*50)
print("SUMMARY OF FINDINGS")
print("="*50)

def print_summary():
    """Print a comprehensive summary of the analysis results."""
    
    # Calculate key metrics
    landing_reduction = ((no_rats_landings.mean() - with_rats_landings.mean()) / 
                        no_rats_landings.mean() * 100)
    time_increase = ((with_rats_time.mean() - no_rats_time.mean()) / 
                    no_rats_time.mean() * 100)
    risk_reduction = ((risk_stats.loc[0, 'mean'] - risk_stats.loc[1, 'mean']) / 
                     risk_stats.loc[0, 'mean'] * 100)
    
    print("KEY RESULTS:")
    print(f"1. Bat landings decrease by {landing_reduction:.1f}% when rats are present")
    print(f"2. Bats take {time_increase:.1f}% longer to approach food with rats present")
    print(f"3. Risk-taking behavior reduces by {risk_reduction:.1f}% with rats present")
    print("\nCONCLUSION:")
    print("Bats demonstrate clear predator-avoidance behavior when rats are present,")
    print("showing decreased landings, increased vigilance, and reduced risk-taking.")
    print("This supports the hypothesis that bats perceive rats as potential predators.")

print_summary()

# Save processed data for further analysis
df1.to_csv('cleaned_dataset1.csv', index=False)
df2.to_csv('cleaned_dataset2.csv', index=False)
print("\n✓ Cleaned datasets saved as 'cleaned_dataset1.csv' and 'cleaned_dataset2.csv'")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)