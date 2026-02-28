import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. PREPARE THE DATA (Real + Synthetic)
# ---------------------------------------------------------
# We use the same generation logic to get enough data for ML
data = []
np.random.seed(42)

# Simulating 500 accidents in Kigali
for i in range(500):
    # Feature generation logic
    # Lat/Long: Center of Kigali with small variations
    lat = -1.9441 + np.random.normal(0, 0.02)
    lon = 30.0619 + np.random.normal(0, 0.02)
    
    # Time: Random hour between 0 and 23
    hour = np.random.randint(0, 24)
    
    # Road Type: Random choice
    road_type = np.random.choice(['Arterial', 'Intersection', 'Residential'], p=[0.5, 0.3, 0.2])
    
    # TARGET LOGIC (The "Truth" we want the AI to learn)
    # If it's Night (hour > 18) AND an Intersection -> High chance of Fatal
    if hour > 18 and road_type == 'Intersection':
        severity = 'Fatal'
    # If it's Day and Residential -> likely Minor
    elif hour < 18 and road_type == 'Residential':
        severity = 'Minor'
    else:
        severity = 'Serious'
        
    data.append([lat, lon, hour, road_type, severity])

df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Hour', 'Road_Type', 'Severity'])

# 2. DATA PREPROCESSING (Converting Text to Numbers)
# ---------------------------------------------------------
# ML models cannot understand text like "Intersection". We encode them.
le_road = LabelEncoder()
df['Road_Type_Code'] = le_road.fit_transform(df['Road_Type'])

# Inputs (X) and Target (y)
X = df[['Latitude', 'Longitude', 'Hour', 'Road_Type_Code']]
y = df['Severity']

# Split Data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN THE AI MODEL
# ---------------------------------------------------------
# We use Random Forest (Standard for tabular classification)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. TEST THE MODEL
# ---------------------------------------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"--- Model Accuracy: {accuracy*100:.2f}% ---")
print("\nDetailed Report:")
print(classification_report(y_test, predictions))

# 5. MAKE A REAL PREDICTION (The "AI" part)
# ---------------------------------------------------------
print("\n--- PREDICTING NEW ACCIDENT ---")
# Scenario: An accident happens at coordinates -1.95, 30.10 at 20:00 (8 PM) on an Intersection.
# The model must guess: Is it Fatal, Serious, or Minor?

new_lat = -1.95
new_lon = 30.10
new_hour = 20 # 8 PM
new_road_type = le_road.transform(['Intersection'])[0] # Convert text to code

prediction = model.predict([[new_lat, new_lon, new_hour, new_road_type]])
print(f"Scenario: Accident at Intersection, 8 PM.")
print(f"AI Prediction: The accident is likely **{prediction[0]}**.")