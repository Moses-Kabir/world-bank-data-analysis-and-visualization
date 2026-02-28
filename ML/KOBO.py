import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np

# ==========================================
# 1. LOAD DATA DIRECTLY
# ==========================================
csv_data = """Province,district,sector,Current mode,Predicted mode
kigali city,nyarugenge,nyarugenge,motorcycles,Public bus
eastern province,NA,NA,Public bus,Public bus
kigali city,nyarugenge,nyarugenge,motorcycles Public bus,Cable car
kigali city,kicukiro,gatenga,Bicycle,Cable car
kigali city,gasabo,NA,motorcycles,Carpool
kigali city,nyarugenge,nyarugenge,Public bus,NA
southern province,NA,NA,motorcycles,Cable car
kigali city,kicukiro,gatenga,motorcycles,Public bus
kigali city,gasabo,remera,motorcycles,Public bus
eastern province,NA,NA,motorcycles,Public bus
southern province,NA,NA,Public bus,Cable car
eastern province,NA,NA,Public bus,Cable car
kigali city,nyarugenge,nyarugenge,motorcycles,Public bus
kigali city,kicukiro,gatenga,motorcycles Public bus,Public bus
kigali city,kicukiro,gatenga,motorcycles Public bus,Public bus
kigali city,nyarugenge,nyarugenge,motorcycles private car,Carpool
kigali city,kicukiro,NA,motorcycles Public bus,Carpool
kigali city,nyarugenge,nyarugenge,motorcycles,Public bus"""

# Read the data
df = pd.read_csv(io.StringIO(csv_data))

# ==========================================
# 2. THE FIX (CRITICAL STEP)
# ==========================================
# This command fills ALL missing empty cells (NaN/Float) with the text "Unknown"
# This prevents "TypeError: float is not iterable"
df.fillna("Unknown", inplace=True)

# Ensure everything is a string
df = df.astype(str)

# ==========================================
# 3. DATA CLEANING
# ==========================================
# Clean up text (Capitalize)
cols_to_clean = ['Province', 'district', 'sector', 'Current mode', 'Predicted mode']
for col in cols_to_clean:
    df[col] = df[col].str.title().str.strip()

# Simplify Categories
def simplify_mode(text):
    # Safety check: if it's still not a string for some reason, convert it
    if not isinstance(text, str):
        return "Unknown"
        
    # Standard cleaning logic
    if text == "Unknown" or text == "Na" or text == "Nan":
        return "Unknown"
    
    # Identify mixed modes (e.g., "Motorcycles Public Bus")
    if " " in text and text not in ["Public Bus", "Cable Car", "Private Car", "Carpool", "Bicycle", "Motorcycles"]:
        return "Multi-Modal (Mixed)"
    
    return text

df['Current mode'] = df['Current mode'].apply(simplify_mode)
df['Predicted mode'] = df['Predicted mode'].apply(simplify_mode)

print("✅ Data Loaded and Cleaned Successfully!")
print(df.head())

# ==========================================
# 4. VISUALIZATION 1: THE SHIFT (Bar Chart)
# ==========================================
plt.figure(figsize=(12, 6))

# Prepare data for plotting
melted = df.melt(id_vars=['Province'], value_vars=['Current mode', 'Predicted mode'], 
                 var_name='Timeline', value_name='Transport Mode')

# Filter out "Unknown" so the chart looks nice
melted_clean = melted[melted['Transport Mode'] != 'Unknown']

sns.countplot(data=melted_clean, x='Transport Mode', hue='Timeline', palette='viridis')
plt.title('Transport Modal Shift: Current vs. Predicted', fontsize=14, fontweight='bold')
plt.xlabel('Transport Mode')
plt.ylabel('Count of Users')
plt.xticks(rotation=45)
plt.legend(title='Scenario')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 5. VISUALIZATION 2: TRANSITION HEATMAP
# ==========================================
plt.figure(figsize=(10, 8))

# Create matrix (Who is moving to where?)
transition = pd.crosstab(df['Current mode'], df['Predicted mode'])

sns.heatmap(transition, annot=True, fmt='d', cmap='Oranges', linewidths=1, linecolor='gray')
plt.title('Commuter Flow: Where are people moving to?', fontsize=14)
plt.ylabel('Current Mode (Origin)')
plt.xlabel('Predicted Mode (Destination)')
plt.tight_layout()
plt.show()

# ==========================================
# 6. AUTOMATED INSIGHTS
# ==========================================
print("\n" + "="*40)
print("       📊 AUTOMATED ANALYSIS RESULTS")
print("="*40)

# Calculate most popular future mode (excluding Unknown)
future_modes = df[df['Predicted mode'] != 'Unknown']['Predicted mode']
if not future_modes.empty:
    top_future = future_modes.value_counts().idxmax()
    top_count = future_modes.value_counts().max()
    print(f"🏆 WINNER: The most desired future transport is '{top_future}' ({top_count} votes).")

# Calculate Cable Car interest
cable_users = len(df[df['Predicted mode'] == 'Cable Car'])
print(f"🚠 CABLE CAR: {cable_users} people specifically requested Cable Cars.")

# Calculate Motorcycle decline
moto_now = len(df[df['Current mode'] == 'Motorcycles'])
moto_future = len(df[df['Predicted mode'] == 'Motorcycles'])
drop = moto_now - moto_future

if drop > 0:
    print(f"📉 MOTO DECLINE: {drop} people want to stop using Motorcycles.")
else:
    print(f"📈 MOTO STABLE: Motorcycle usage is expected to rise or stay same.")