import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. CREATE IDEAL FULL INPUT (Synthetic Data)
# ==========================================
np.random.seed(42) # Ensures you get the exact same results as me
n_employees = 1000

# Feature 1: Satisfaction Level (0.0 to 1.0)
satisfaction = np.random.uniform(0.1, 1.0, n_employees)

# Feature 2: Years at Company (2 to 10)
years = np.random.randint(2, 10, n_employees)

# Feature 3: Average Monthly Hours (130 to 310)
hours = np.random.normal(200, 30, n_employees).astype(int)

# Feature 4: Salary Level (0=Low, 1=Medium, 2=High)
salary_numeric = np.random.choice([0, 1, 2], n_employees, p=[0.4, 0.4, 0.2])

# -- LOGIC FOR THE TARGET VARIABLE ('Left') --
# We create a pattern: People leave if Satisfaction is low OR they work too many hours
# We add some randomness so the model has to "learn" instead of just memorizing
base_prob = np.zeros(n_employees)
base_prob[satisfaction < 0.5] += 0.6   # Low satisfaction increases chance of leaving
base_prob[hours > 250] += 0.4          # Overworked increases chance of leaving
base_prob[salary_numeric == 2] -= 0.3  # High salary reduces chance of leaving

# Convert probabilities to 0 (Stay) or 1 (Left)
left = (np.random.rand(n_employees) < base_prob).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Satisfaction': satisfaction,
    'YearsAtCompany': years,
    'MonthlyHours': hours,
    'SalaryLevel': salary_numeric,
    'Left': left
})

print("--- DATA SNAPSHOT (First 5 Rows) ---")
print(df.head())
print("\n")

# ==========================================
# 2. PREPARE & SPLIT DATA
# ==========================================
# Inputs (X) and Target (y)
X = df.drop('Left', axis=1)
y = df['Left']

# Split: 80% for Training (Learning), 20% for Testing (Exam)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. BUILD & TRAIN THE MODEL
# ==========================================
# Random Forest is powerful and requires little tuning
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. PREDICT & EVALUATE (The Results)
# ==========================================
predictions = model.predict(X_test)

print("--- MODEL PERFORMANCE RESULTS ---")
print(f"Accuracy Score: {accuracy_score(y_test, predictions)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ==========================================
# 5. VISUALIZE RESULTS
# ==========================================
plt.figure(figsize=(12, 5))

# Plot A: Confusion Matrix (Did we predict correctly?)
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix\n(Accurate Guesses vs Errors)')
plt.xlabel('Predicted (0=Stay, 1=Left)')
plt.ylabel('Actual (0=Stay, 1=Left)')

# Plot B: Feature Importance (Why did they leave?)
plt.subplot(1, 2, 2)
importance = model.feature_importances_
features = X.columns
sns.barplot(x=importance, y=features, palette='viridis')
plt.title('What Drives Employee Churn?\n(Feature Importance)')
plt.xlabel('Importance Score')

plt.tight_layout()
plt.show()