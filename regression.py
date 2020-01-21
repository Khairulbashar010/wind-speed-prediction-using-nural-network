# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
df = pd.read_csv(r"please\specify\the\path\to\weather_data.csv") #SPECIFY THE PATH TO THE DATASET
df = df.dropna()
del df['timestamp']
X = df.iloc[:30000, :7].values
y = df.iloc[:30000, 7].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = y.reshape(-1, 1)
y = sc_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting the Regression Model to the dataset
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)


from sklearn.svm import SVR
regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
regressor.fit(X_train, y_train)
# Create your regressor here
#y_test = y_test.reshape(-1,1)
# Predicting a new result



y_pred = regr.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
y_pred
y_test = sc_y.inverse_transform(y_test)

y_r_pred = regressor.predict(X_test)
y_r_pred = sc_y.inverse_transform(y_pred)





regr.score(X_test, y_test)
regressor.score(X_test, y_test)



# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
y_train = y_train.ravel()
# Train the model on training data
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)
errors = abs(predictions - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
#accuracy
print('Accuracy:', round(accuracy, 2), '%.')