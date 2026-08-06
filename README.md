# house_price_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Step 1: Load Dataset ----------------
# Replace with your Kaggle dataset file (e.g., "house_prices.csv")
data = pd.read_csv("house_prices.csv")

print("✅ Data Loaded Successfully!")
print(data.head())

# ---------------- Step 2: Preprocessing ----------------
# Handle missing values
data = data.dropna()

# Encode categorical variables (like location)
data = pd.get_dummies(data, drop_first=True)

# Select features (adjust based on your dataset columns)
features = ["Rooms", "Size", "Location_NewYork", "Location_London"]  # Example
X = data[features]
y = data["Price"]

# ---------------- Step 3: Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Step 4: Train Model ----------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- Step 5: Evaluate Model ----------------
y_pred = model.predict(X_test)

print("\n📊 Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ---------------- Step 6: Visualization ----------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ---------------- Step 7: Prediction Example ----------------
sample_house = np.array([[3, 1200, 1, 0]])  # Example: 3 rooms, 1200 sqft, NewYork=1, London=0
predicted_price = model.predict(sample_house)
print("\n💡 Predicted Price for Sample House:", predicted_price[0])
