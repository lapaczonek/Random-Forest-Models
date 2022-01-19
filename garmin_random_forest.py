from pyexpat import features
from random import random
from statistics import mean
from tkinter.tix import Tree
import pandas as pd
import matplotlib as plt
import numpy as np

data = pd.read_csv('garmin.csv')
data['Average Stress'] = ['23.33','23.33','23.33','23.33','23.33','23.33','23.33','23.33','23.33','23.33','23.33','23.33']
data.to_csv('garmin.csv', index= False)
print(data.head())
#Change the data for Monday-Friday to binary
data = pd.get_dummies(data)
print(data)
print(data.isna())
#Changing null values to the average 
data = data.fillna(data.mean())
data = data.fillna(data.mean())
print(data.isna())
#Identifying anomalies/ null
print("shape of the data is: ", data.shape)
print(data.describe())

#Display first 5 columns of the last 12 columns 
print(data.iloc[:, 5:].head(5))

#What we want to predict
predictor_variable = np.array(data['Stress'])
data = data.drop('Stress', axis=1)
print(data.columns)
#Saves the data names
data_list = list(data.columns)

print(data_list)
#Converts to numpy array
data = np.array(data)
print(data)

#Trains algorithim
from sklearn.model_selection import train_test_split
train_data, test_data, train_predictor_variable, test_predictor_variable = train_test_split(data, predictor_variable, test_size=0.40, random_state=42)

#Make sure the data for training predictor column matches the data training columns
print("Training Data shape:", train_data.shape)
print("Training Predictor Variable:", train_predictor_variable.shape)
print("Testing Data shape:", test_data.shape)
print("Testing Predictor Variable:", test_predictor_variable.shape)

#Error we should beat the average
baseline_prediction = test_data[:, data_list.index('Average Stress_23.33')]
print(baseline_prediction)
print(predictor_variable)
baseline_errors = abs(baseline_prediction - test_predictor_variable)
print("Baseline error ", round(np.mean(baseline_errors), 2))

#Train Model
from sklearn.ensemble import RandomForestRegressor


#Initiate with 1000 decision trees
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)
#Train the Model on training data
rf.fit(train_data, train_predictor_variable)

#Predict the outcomes
predictions = rf.predict(test_data)
print("\n\n", predictions)

errors = abs(predictions - test_predictor_variable)
print("Mean Absolute Error: ", round(np.mean(errors), 2), "degrees")

#Calculate Mean Absolute percentage error (MAPE)
mape = 100 * (errors / test_predictor_variable)

#CAlculate accuracy
accuracy = 100 - np.mean(mape)
print("Accuracy is : ", round(accuracy), 2,"%")

#Visualizing one tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import tree

print(rf.estimators_)
print(len(rf.estimators_))
feature=data_list
class_name = predictor_variable
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[1], feature_names = feature, class_names=class_name, filled = True);
fig.savefig('rf_one_decision_tree.png')
pyplot.show()

#Checking to see which variables are more important 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_data, train_predictor_variable)
importance = model.coef_
for i,j in enumerate(importance):
    i = data_list[i]
    print(f'Variable: {i} , Score: {j}')


x_values = list(range(len(importance)))
plt.bar(x_values, importance, orientation= 'vertical')
plt.xticks(x_values, data_list, rotation= 'vertical')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()


# Get numerical feature importances
importances = list(rf.feature_importances_)
print()
print(importances)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(data_list, importances)]
print()
print(feature_importances)

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('\n\n Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=None)
# Extract the two most important variables

important_indices = [data_list.index("Yoga"), data_list.index('Intensity Min'), data_list.index('day'), data_list.index('Calories')]
print(important_indices)
train_important = train_data[:, important_indices]
test_important = test_data[:, important_indices]
print(train_important.shape)
print("\n\n\n\n\n")
print(test_important.shape)

# Train the random forest
rf_most_important.fit(train_important, train_predictor_variable)

# Make predictions and determine the error
print(test_important)
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_predictor_variable)
print(predictions.shape)
#Display the performance metrics
print()
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_predictor_variable))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

#Visualizing the data
x_values = list(range(len(importances)))
#Make bar chart
plt.bar(x_values, importances, orientation= 'vertical')
plt.xticks(x_values, data_list, rotation= 'vertical')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()

print()

print(data)

print(data_list)

#Plotting data along with our predictions on a line graph
import datetime

#get dates of training variables
months = data[:, data_list.index('month')]
days = data[:, data_list.index('day')]
years = data[:, data_list.index('year')]
dates = [str(int(year)) + '-' + str(int(month))+ '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

#True data
true_data = pd.DataFrame(data = {'date': dates, 'actual': predictor_variable})

# Dates of predictions
months = test_data[:, data_list.index('month')]
days = test_data[:, data_list.index('day')]
years = test_data[:, data_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

# Dataframe with predictions and dates
print(predictions)
print(test_dates)
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})

# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()

# Graph labels
plt.xlabel('Date')
plt.ylabel('AVG Stress per Day')
plt.title('Actual and Predicted Values')
plt.show()

