# -*- coding: utf-8 -*-
"""House Price prediction using Linear regression

No preprocessing + Linear regression
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the datasets
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# Select relevant features and target variable
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X = train_data[features]
y = train_data[target]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate the R² on the validation set
r2 = r2_score(y_val, y_pred)
print(f"R²: {r2}")

# Select the same features from the test set
X_test = test_data[features]

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

# Save the submission DataFrame to a CSV file
submission_path = './house_price_predictions.csv'
submission.to_csv(submission_path, index=False)

print("Predictions saved to:", submission_path)

"""Simple Linear regression + preprocessing"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the datasets
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# Select relevant features and target variable
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# Check for duplicated values and drop them
train_data = train_data.drop_duplicates()

# Fill missing values with the mean for both train and test data
for feature in features:
    train_data[feature].fillna(train_data[feature].mean(), inplace=True)
    test_data[feature].fillna(test_data[feature].mean(), inplace=True)

# Extract features and target variable
X = train_data[features]
y = train_data[target]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training and validation sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Calculate the R² score for both training and validation sets
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"Training R²: {r2_train}")
print(f"Validation R²: {r2_val}")

# Select the same features from the test set
X_test = test_data[features]

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

# Save the submission DataFrame to a CSV file
submission_path = './house_price_predictions2.csv'
submission.to_csv(submission_path, index=False)

print("Predictions saved to:", submission_path)

"""Ridge Technique + preprocessing + scaling + using all columns"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Load the datasets
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# Select relevant features and target variable
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'GarageArea']
target = 'SalePrice'

# Check for duplicated values and drop them
train_data = train_data.drop_duplicates()

# Fill missing values with the mean for both train and test data
for feature in features:
    train_data[feature].fillna(train_data[feature].mean(), inplace=True)
    test_data[feature].fillna(test_data[feature].mean(), inplace=True)

# Extract features and target variable
X = train_data[features]
y = train_data[target]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data[features])

# Create and train the Ridge regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Make predictions on the training and validation sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Calculate the R² score for both training and validation sets
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"Training R²: {r2_train}")
print(f"Validation R²: {r2_val}")

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

# Save the submission DataFrame to a CSV file
submission_path = './house_price_predictions3.csv'
submission.to_csv(submission_path, index=False)

print("Predictions saved to:", submission_path)

"""More improved Linear regression model"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the datasets
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# Select relevant features and target variable
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual', 'YearBuilt', 'TotalBsmtSF']
target = 'SalePrice'

# Check for duplicated values and drop them
train_data = train_data.drop_duplicates()

# Fill missing values with the mean for both train and test data
for feature in features:
    train_data[feature].fillna(train_data[feature].mean(), inplace=True)
    test_data[feature].fillna(test_data[feature].mean(), inplace=True)

# Extract features and target variable
X = train_data[features]
y = train_data[target]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training and validation sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Calculate the R² score for both training and validation sets
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"Training R²: {r2_train}")
print(f"Validation R²: {r2_val}")

# Select the same features from the test set
X_test = test_data[features]

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

# Save the submission DataFrame to a CSV file
submission_path = './house_price_predictions4.csv'
submission.to_csv(submission_path, index=False)

print("Predictions saved to:", submission_path)