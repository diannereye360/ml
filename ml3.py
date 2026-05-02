from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
 
 
dataset = load_dataset("scikit-learn/auto-mpg")
train_data = dataset["train"]
 
data_dict = train_data.to_dict()
print(data_dict.keys())
 
weight = data_dict['weight']
mpg = data_dict['mpg']
 
# Scatter plot: Weight vs MPG
plt.scatter(weight, mpg, alpha=0.5, color='steelblue')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Weight vs MPG')
plt.tight_layout()
plt.show()
 
# Print each row
num_rows = len(mpg)
for i in range(num_rows):
  row = [
    data_dict['cylinders'][i],
    data_dict['displacement'][i],
    int(data_dict['horsepower'][i] if data_dict['horsepower'][i] != '?' else 0),
    data_dict['acceleration'][i],
    data_dict['model year'][i]
  ]
  print(f"Row {i}: {row}")
 
# Convert to DataFrame
df = pd.DataFrame(data_dict)
 
# Keep numeric columns only (drop car name)
df_numeric = df.drop(columns=['car name']).apply(pd.to_numeric, errors='coerce')
df_numeric = df_numeric.dropna()
 
# Train test split
X = df_numeric.drop(columns=['mpg'])
y = df_numeric['mpg']
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=11)
 
# Train model
model = LinearRegression()
model.fit(X=x_train, y=y_train)
 
predicted = model.predict(X=x_test)
 
# Scatter plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predicted, alpha=0.4)
plt.title('Actual vs Predicted MPG')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.tight_layout()
plt.show()
 
print("\n Sample Predictions:")
for i in range(10):
  print(f"Actual: {y_test.iloc[i]:.2f}, Predicted: {predicted[i]:.2f}")