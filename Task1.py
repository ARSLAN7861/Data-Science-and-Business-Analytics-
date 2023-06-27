#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read data from the remote link
data = pd.read_csv('http://bit.ly/w-data')

# Extract the input (hours) and output (percentage) variables
hours = data['Hours'].values
percentage = data['Scores'].values

# Reshape the input variable to a 2D array
hours = hours.reshape(-1, 1)

# Split the data into training and testing sets (80% training, 20% testing)
hours_train, hours_test, percentage_train, percentage_test = train_test_split(hours, percentage, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(hours_train, percentage_train)

# Make predictions on the testing data
predictions = model.predict(hours_test)

# Calculate the mean squared error
mse = mean_squared_error(percentage_test, predictions)

# Calculate the coefficient of determination (R-squared)
r2 = r2_score(percentage_test, predictions)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict the percentage for a new student
new_hours = np.array([[7.78]])  # Number of hours the new student studies
predicted_percentage = model.predict(new_hours)

print("Predicted Percentage:", predicted_percentage)


# In[ ]:




