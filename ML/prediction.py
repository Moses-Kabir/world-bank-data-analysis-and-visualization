import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. GENERATE REALISTIC KIGALI TRAFFIC DATA
# ==========================================
# We simulate 2,000 traffic reports across different hotspots
np.random.seed(42)
n_samples = 2000

# -- Features --
# Locations: specific hotspots in Kigali
locations = np.random.choice(
    ['Kimironko', 'Remera (Giporoso)', 'CBD (Town)', 'Nyabugogo', 'Kacyiru', 'Kanombe'], 
    n_samples
)

# Time: Hour of the day (0 to 23)
hours = np.random.randint(0, 24, n_samples)

# Day: 0=Monday ... 6=Sunday
days = np.random.randint(0, 7, n_samples)

# Weather: Rain creates massive jams in Kigali
weather = np.random.choice(['Clear', 'Rainy', 'Foggy'], n_samples, p=[0.7, 0.2, 0.1])

# Special Event: (e.g., Conference at BK Arena or Convention Center)
event = np.random.choice([0, 1], n_samples, p=[0.9, 0.1]) # 10% chance of event

# -- LOGIC FOR TARGET VARIABLE (Congestion Level 0-100) --
congestion = []

for i in range(n_samples):
    base_jam = 20  # Minimum traffic exists
    
    # 1. Location Logic
    if locations[i] == 'Nyabugogo': base_jam += 30 # Always busy
    elif locations[i] == 'Remera (Giporoso)': base_jam += 20
    elif locations[i] == 'CBD (Town)': base_jam += 25
    
    # 2. Time Logic (Rush Hours: 7-9 AM and 5-7 PM)
    if 7 <= hours[i] <= 9: base_jam += 35
    elif 17 <= hours[i] <= 19: base_jam += 40
    elif 0 <= hours[i] <= 5: base_jam -= 15 # Night is empty
    
    # 3. Weather Logic
    if weather[i] == 'Rainy': base_jam += 25 # Rain = Chaos
    
    # 4. Weekend Logic
    if days[i] >= 5: base_jam -= 20 # Weekends are lighter
    
    # 5. Event Logic
    if event[i] == 1: base_jam += 15
    
    # Add Random Noise (Real life is unpredictable)
    noise = np.random.randint(-5, 5)
    total = base_jam + noise
    
    # Clip logic (Can't be <0 or >100)
    total = np.clip(total, 0, 100)
    congestion.append(total)

# Create DataFrame
df = pd.DataFrame({
    'Location': locations,
    'Hour': hours,
    'Day_of_Week': days,
    'Weather': weather,
    'Special_Event': event,
    'Congestion_Level': congestion
})

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# Machines only understand numbers. We encode text.
le_loc = LabelEncoder()
df['Location_Code'] = le_loc.fit_transform(df['Location'])

le_weather = LabelEncoder()
df['Weather_Code'] = le_weather.fit_transform(df['Weather'])

# Features (X) and Target (y)
X = df[['Location_Code', 'Hour', 'Day_of_Week', 'Weather_Code', 'Special_Event']]
y = df['Congestion_Level']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. TRAIN THE MODEL (Random Forest)
# ==========================================
# We use 100 trees to vote on the result
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATE RESULTS
# ==========================================
preds = model.predict(X_test)

print("--- MODEL ACCURACY REPORT ---")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, preds):.2f}") 
# MAE means: "On average, our guess is only off by X percent."
print(f"R2 Score (Fit Quality): {r2_score(y_test, preds)*100:.2f}%")
print("-" * 30)

# ==========================================
# 5. MAKE AN IDEAL PREDICTION (Interactive)
# ==========================================
# Let's predict traffic for a specific scenario
# SCENARIO: 6:00 PM (18:00), raining in Remera (Giporoso) on a Tuesday
loc_input = 'Remera (Giporoso)'
hour_input = 18
day_input = 1 # Tuesday
weather_input = 'Rainy'
event_input = 0 # No special event

# Convert inputs to codes
loc_code = le_loc.transform([loc_input])[0]
weather_code = le_weather.transform([weather_input])[0]

# Predict
predicted_jam = model.predict([[loc_code, hour_input, day_input, weather_code, event_input]])

print(f"\n🔮 PREDICTION RESULT:")
print(f"Scenario: {loc_input} at {hour_input}:00, Weather: {weather_input}")
print(f"Predicted Congestion Level: {predicted_jam[0]:.1f}/100")
if predicted_jam[0] > 75:
    print("⚠️ ADVICE: Avoid this route! Heavy Jam expected.")
elif predicted_jam[0] > 50:
    print("🚗 ADVICE: Moderate traffic. Expect delays.")
else:
    print("✅ ADVICE: Road is clear.")

# ==========================================
# 6. VISUALIZE IMPORTANCE
# ==========================================
plt.figure(figsize=(10, 5))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.plot(kind='barh', color='teal')
plt.title('What Causes Traffic in Kigali?')
plt.xlabel('Importance Score')
plt.show()