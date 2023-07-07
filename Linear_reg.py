from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score 

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler 

  

# Assuming you have already preprocessed your dataset and have features (X) and labels (y) 

  

# Split the dataset into training and testing subsets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

  

# Perform feature scaling on the training data 

scaler = StandardScaler() 

X_train_scaled = scaler.fit_transform(X_train) 

  

# Create an instance of the logistic regression model 

logistic_regression = LogisticRegression() 

  

# Train the logistic regression model on the scaled training data 

logistic_regression.fit(X_train_scaled, y_train) 

  

# Perform feature scaling on the testing data 

X_test_scaled = scaler.transform(X_test) 

  

# Make predictions on the testing data 

y_pred = logistic_regression.predict(X_test_scaled) 

  

# Calculate the accuracy of the model 

accuracy = accuracy_score(y_test, y_pred) 

print("Accuracy: ", accuracy) 
