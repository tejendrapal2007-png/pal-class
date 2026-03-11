# ==============================
# Student Marks Prediction
# Linear Regression (Full Code)
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------
# Step 1: Create Dataset
# ------------------------------
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [30, 35, 45, 50, 60, 65, 70, 80, 85, 95]
}

df = pd.DataFrame(data)

print("Dataset:\n")
print(df)

# ------------------------------
# Step 2: Define Features & Target
# ------------------------------
X = df[["Hours"]]   # Independent variable
y = df["Marks"]     # Dependent variable

# ------------------------------
# Step 3: Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Step 4: Apply Linear Regression
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 5: Predictions
# ------------------------------
y_pred = model.predict(X_test)

print("\nPredicted Marks:", y_pred)

# ------------------------------
# Step 6: Model Evaluation
# ------------------------------
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ------------------------------
# Step 7: Predict New Value
# ------------------------------
hours = np.array([[9]])  # Student studied 9 hours
predicted_marks = model.predict(hours)

print("\nPredicted marks for 9 hours study:", predicted_marks[0])

# ------------------------------
# Step 8: Print Equation
# ------------------------------
print("\nModel Equation:")
print("Marks =", model.coef_[0], "* Hours +", model.intercept_)

# ------------------------------
# Step 9: Plot Graph
# ------------------------------
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Hours vs Marks Prediction")
plt.show()