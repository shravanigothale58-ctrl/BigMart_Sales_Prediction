# BigMart Sales Prediction Project
# Beginner-friendly project using the uploaded bigmart.csv dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
# STEP 1: Load the CSV file.
# Keep bigmart.csv in the same folder as this Python file.

df = pd.read_csv("bigmart.csv")
# STEP 2: Display a few rows to understand the dataset.
print("First five records:")
print(df.head())
print("\nDataset shape (rows, columns):")
print(df.shape)
# STEP 3: Select the input feature and target column.
# Item_MRP is the product's maximum retail price.
# Item_Outlet_Sales is the sales value we want to predict.
X = df[["Item_MRP"]]
y = df["Item_Outlet_Sales"]
# STEP 4: Split the data.
# 80% is used for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# STEP 5: Create the Linear Regression model.
model = LinearRegression()
# STEP 6: Train the model using the training data.
model.fit(X_train, y_train)
# STEP 7: Predict sales for the test data.
y_pred = model.predict(X_test)
# STEP 8: Evaluate the model.
# MAE tells us the average absolute prediction error in sales units.
mae = mean_absolute_error(y_test, y_pred)
# R-squared indicates how well the model explains target variation.
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R-squared score:", round(r2, 3))
# STEP 9: Predict sales for a new item MRP.
# Change 150 to test a different price.
new_item = pd.DataFrame({"Item_MRP": [150]})
predicted_sales = model.predict(new_item)
print("\nNew Item MRP:", 150)
print("Predicted Item Outlet Sales:", round(predicted_sales[0], 2))
# STEP 10: Plot actual sales against item MRP.
plt.scatter(
X_test["Item_MRP"],
y_test,
label="Actual Sales"
)
# Sort the test rows so the prediction line is drawn neatly.
plot_data = pd.DataFrame({
"Item_MRP": X_test["Item_MRP"].to_numpy(),
"Predicted_Sales": y_pred
}).sort_values("Item_MRP")

plt.plot(
plot_data["Item_MRP"],
plot_data["Predicted_Sales"],
label="Regression Line",
)
# Add a title and labels to explain the graph.
plt.title("BigMart Sales Prediction")
plt.xlabel("Item MRP")
plt.ylabel("Item Outlet Sales")
plt.legend()
plt.grid(True)
# Show the graph.
plt.show()