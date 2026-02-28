import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('D:\TRANSPORTATION ENGINEERING\AUTOSAVE\WB_RISE_EE_TRN.csv')

# 1. CLEANING
# Keep only necessary columns
cols_to_keep = ['REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']
df_clean = df[cols_to_keep].copy()

# Rename for clarity
df_clean.columns = ['Country', 'Year', 'Score']

# Convert types
df_clean['Score'] = pd.to_numeric(df_clean['Score'], errors='coerce')
df_clean['Year'] = df_clean['Year'].astype(int)

# Drop any rows with missing scores (if any)
df_clean = df_clean.dropna(subset=['Score'])

# 2. ANALYSIS: Global Trend over Time
global_trend = df_clean.groupby('Year')['Score'].mean().reset_index()

# 3. ANALYSIS: Top Performers in 2023
latest_year = df_clean[df_clean['Year'] == 2023].sort_values(by='Score', ascending=False)
top_10_2023 = latest_year.head(10)
bottom_10_2023 = latest_year[latest_year['Score'] > 0].tail(10) # Excluding zeros for focus

# --- VISUALIZATION ---
plt.figure(figsize=(14, 10))

# Plot 1: Global Average Trend
plt.subplot(2, 1, 1)
sns.lineplot(data=global_trend, x='Year', y='Score', marker='o', color='teal', linewidth=2.5)
plt.title('Global Average RISE Score: Transport Sector (2010–2023)', fontsize=14)
plt.ylabel('Average Score (0-100)')
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Top 10 Countries in 2023
plt.subplot(2, 1, 2)
sns.barplot(data=top_10_2023, x='Score', y='Country', palette='viridis')
plt.title('Top 10 Countries by Transport Sector Score (2023)', fontsize=14)
plt.xlabel('RISE Score')

plt.tight_layout()
plt.show()

# 4. ANALYSIS: Progress Summary
pivot_df = df_clean.pivot(index='Country', columns='Year', values='Score')
pivot_df['Improvement'] = pivot_df[2023] - pivot_df[2010]
most_improved = pivot_df['Improvement'].sort_values(ascending=False).head(5)

print("--- Data Summary ---")
print(f"Total countries analyzed: {df_clean['Country'].nunique()}")
print(f"Global Average Score in 2010: {global_trend.iloc[0]['Score']:.2f}")
print(f"Global Average Score in 2023: {global_trend.iloc[-1]['Score']:.2f}")
print("\nMost Improved Countries (2010 vs 2023):")
print(most_improved)