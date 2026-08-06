import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv("Housing.csv")
X = pd.get_dummies(df.drop('price', axis=1), drop_first=True)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"R-squared Score: {r2_score(y_test, predictions):.2f}")
print(f"Mean Absolute Error: ${mean_absolute_error(y_test, predictions):,.2f}")
plt.figure(figsize=(9, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='teal', edgecolor='black')
max_val = max(y_test.max(), predictions.max())
min_val = min(y_test.min(), predictions.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
plt.title('Linear Regression: Actual vs. Predicted House Prices', fontsize=14)
plt.xlabel('Actual Price ($)', fontsize=12)
plt.ylabel('Predicted Price ($)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

