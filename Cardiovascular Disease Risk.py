#!/usr/bin/env python
# coding: utf-8

# # Step 1: Data Loading and Exploration

# In[1]:


import pandas as pd

# Load the dataset
file_path = 'cardio_train.csv'
data = pd.read_csv(file_path, sep=';')

# Display the first few rows of the dataset
print(data.head())


# ## Explore the structure, dimensions, missing values, and class distribution.

# In[2]:


# Check dataset structure and basic info
data.info()

# Check dimensions
print(f"Dataset dimensions: {data.shape}")

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

# Check class distribution (assuming 'cardio' is the label column)
class_distribution = data['cardio'].value_counts()
print("Class distribution:\n", class_distribution)


# ## Visualize basic statistics and relationships.

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Summary statistics
print(data.describe())

# Pairplot for basic relationships (select relevant columns)
sns.pairplot(data[['age', 'height', 'weight', 'ap high', 'ap low', 'cardio']])
plt.show()


# # Step 2: Data Preprocessing
# 
# Handle missing values (if any).

# In[4]:


# Handle missing values using forward fill
data.ffill(inplace=True)


# #### Feature engineering (e.g., adding BMI feature).

# In[5]:


# Feature engineering: Calculate BMI
# BMI = weight (kg) / (height (m))^2
data['BMI'] = data['weight'] / (data['height']/100) ** 2


# #### Standardize or normalize numerical features.

# In[6]:


from sklearn.preprocessing import StandardScaler

# Define features to scale
numerical_features = ['age', 'height', 'weight', 'ap high', 'ap low', 'BMI']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])


# #### Encode categorical variables (if applicable).

# In[7]:


# Encode categorical features using one-hot encoding
categorical_features = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)


# # Step 3: Exploratory Data Analysis (EDA)
# 
# Visualize the correlations between variables using heatmaps or pairplots.

# In[8]:


# Correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# #### Analyze feature distributions (histograms, boxplots, etc.).

# In[9]:


# Histograms for feature distributions
features_to_plot = ['age', 'height', 'weight', 'ap high', 'ap low', 'BMI']

for feature in features_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.show()


# #### Identify key factors affecting cardiovascular disease risk.

# In[10]:


# Boxplot for key factors
key_factors = ['age', 'BMI', 'ap high', 'ap low']

for factor in key_factors:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='cardio', y=factor, data=data)
    plt.title(f"{factor} by Cardiovascular Disease")
    plt.show()


# # Step 4: Data Splitting
# 
# Split the data into training and testing sets.

# In[11]:


from sklearn.model_selection import train_test_split

# Define features and target variable
X = data.drop('cardio', axis=1)
y = data['cardio']

# Split the data using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Display the shapes of the resulting splits
print(f"Training set size: {X_train.shape}, {y_train.shape}")
print(f"Testing set size: {X_test.shape}, {y_test.shape}")


# # Step 5: Feature Scaling
# 
# Scale the features using StandardScaler or similar methods to prepare for deep learning.

# In[12]:


from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Apply scaling to training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display scaled data shapes
print(f"Scaled Training set size: {X_train_scaled.shape}")
print(f"Scaled Testing set size: {X_test_scaled.shape}")


# # Step 6: Model Building
# 
# Define a deep learning model using Keras/TensorFlow.

# In[13]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()


# # Step 7: Model Training
# 
# Train the model using the training data.

# In[14]:


# Train the model with validation
history = model.fit(
    X_train_scaled, y_train, 
    validation_split=0.2, 
    epochs=50, 
    batch_size=32, 
    verbose=1
)

# Plot training vs. validation loss and accuracy curves
import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # Step 8: Model Evaluation
# 
# Evaluate the trained model on the test data.

# In[15]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = (model.predict(X_test_scaled) > 0.5).astype('int32')

# Generate evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:\n", accuracy_score(y_test, y_pred))


# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# # Step 9: Model Optimization
# 
# Improve model performance using various optimization techniques.

# In[16]:


get_ipython().system('pip install scikeras')


# In[20]:


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a function for creating the model
def create_model(optimizer='adam', dropout_rate=0.3):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(model=create_model, verbose=1)

# Define the parameter grid
param_distributions = {
    'batch_size': [16, 32, 64],
    'epochs': [50, 100],
    'model__optimizer': ['adam', 'rmsprop'],
    'model__dropout_rate': [0.2, 0.3, 0.4]
}


# Perform Randomized Search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, cv=2, n_iter=5, n_jobs=-1, verbose=2, random_state=42)
random_search_result = random_search.fit(X_train_scaled, y_train)

# Display best parameters and performance
print(f"Best Accuracy: {random_search_result.best_score_:.4f}")
print("Best Parameters:", random_search_result.best_params_)


# # Step 10: Model Interpretability
# 
# 

# #### 1. Permutation Feature Importance (scikit-learn)

# In[51]:


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# Define prediction function for binary labels
def predict_binary(X):
    return (model_for_importance.predict(X) > 0.5).astype(int)

# Calculate permutation feature importance
from sklearn.base import BaseEstimator

# Create a custom estimator wrapper
class CustomEstimator(BaseEstimator):
    def fit(self, X, y):
        pass  # No fitting needed for pre-trained models

    def predict(self, X):
        return (model_for_importance.predict(X) > 0.5).astype(int)

# Initialize the custom estimator
custom_estimator = CustomEstimator()

# Re-run permutation importance
result = permutation_importance(
    custom_estimator, X_test_scaled, y_test, n_repeats=5, random_state=42, scoring='accuracy'
)

# Visualize feature importances
sorted_idx = result.importances_mean.argsort()
plt.barh(np.array(X.columns)[sorted_idx], result.importances_mean[sorted_idx])
plt.title("Permutation Feature Importance")
plt.show()


# #### 2. Logistic Regression coefficients

# In[46]:


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Train logistic regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Visualize coefficients
plt.barh(X.columns, lr_model.coef_[0])
plt.title("Logistic Regression Coefficients")
plt.show()


# # Step 11: Integration and Deployment
# 
# Prepare the End-to-End Data Pipeline

# In[63]:


import joblib
import pickle
import os

# Save the trained model in Keras format
model_filename = 'cvd_risk_model.keras'
model_for_importance.save(model_filename)

# Save the scaler
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)


# #### Deploy the Model using Streamlit

# In[ ]:


# app.py
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the model and scaler
model = load_model('cvd_risk_model.keras')
scaler = joblib.load('scaler.pkl')

# Streamlit interface
st.title("Cardiovascular Disease Risk Prediction")

# Create input fields
age = st.number_input("Age", min_value=0, max_value=120, step=1)
height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
weight = st.number_input("Weight (kg)", min_value=20, max_value=200, step=1)
ap_high = st.number_input("Systolic Blood Pressure", min_value=50, max_value=250, step=1)
ap_low = st.number_input("Diastolic Blood Pressure", min_value=30, max_value=150, step=1)
bmi = weight / (height / 100) ** 2

# Prediction button
if st.button("Predict"):
    features = np.array([[age, height, weight, ap_high, ap_low, bmi]])
    scaled_features = scaler.transform(features)
    prediction = (model.predict(scaled_features) > 0.5).astype(int)
    result = "High Risk" if prediction[0][0] == 1 else "Low Risk"
    st.success(f"Predicted Risk: {result}")


# In[ ]:




