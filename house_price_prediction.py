import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("train.csv")

# Remove outliers
df = df[df['GrLivArea'] < 4000]

# Features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Handle missing values
X = X.fillna(X.mean())

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Load test dataset
test_df = pd.read_csv("test.csv")

x_final = test_df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Handle missing values
x_final = x_final.fillna(x_final.mean())

# Final predictions
final_predictions = model.predict(x_final)

# Print predictions
print(final_predictions)

# Metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization 1
plt.scatter(df['GrLivArea'], df['SalePrice'])
plt.xlabel("Ground Living Area (Square Footage)")
plt.ylabel("House Sale Price")
plt.title("Living Area vs House Price")
plt.show()

# Visualization 2
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
