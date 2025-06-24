#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Data Loading and Exploration

# ### Data Loading and Verification
# 
# In the next step, the dataset `CVD_cleaned.csv` will be loaded into a pandas DataFrame to ensure compatibility and processing efficiency. The dataset's structure will be verified by examining its shape, column names, and data types, ensuring it is correctly formatted for analysis. Additionally, libraries such as `matplotlib.pyplot` and `seaborn` will be imported to facilitate exploratory data analysis (EDA) in subsequent steps, preparing the dataset for further processing and insights.
# 

# In[1]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "CVD_cleaned.csv"  
df = pd.read_csv(file_path)  

# Verify the data
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print("Data Types:\n", df.dtypes)


# ### Dataset Overview
# 
# The dataset contains **308,854 rows and 19 columns**, representing a large and comprehensive dataset for analysis. The column names indicate a mix of categorical and numerical variables related to health and lifestyle factors, such as `General_Health`, `Exercise`, `Heart_Disease`, `BMI`, and dietary habits, which are relevant for predicting cardiovascular disease outcomes.
# 
# #### Key Observations:
# 1. **Categorical Data**:
#    - Columns like `General_Health`, `Checkup`, `Exercise`, `Heart_Disease`, `Skin_Cancer`, `Other_Cancer`, and `Sex` are stored as `object` types. These are likely categorical variables and may need encoding (e.g., one-hot or label encoding) for model training.
# 
# 2. **Numerical Data**:
#    - Columns like `Height_(cm)`, `Weight_(kg)`, `BMI`, and dietary consumption metrics (`Alcohol_Consumption`, `Fruit_Consumption`, etc.) are stored as `float64`. These are ready for numerical analysis but may require scaling or normalization.
# 
# 3. **Target Variable**:
#    - `Heart_Disease` is listed as an `object`. This column is likely the target variable and will need conversion to a numerical format (e.g., binary 0/1 encoding).
# 

# ### Target Variable: `Heart_Disease`
# 
# The `Heart_Disease` column is selected as the target variable for the following reasons:
# 
# 1. **Relevance to the Problem Statement**:
#    - The goal of this project is to predict the likelihood of cardiovascular disease. The `Heart_Disease` column explicitly represents whether an individual has the condition (`Yes`) or not (`No`).
# 
# 2. **Binary Classification**:
#    - This column is categorical with two possible outcomes (`Yes` and `No`). Its binary nature aligns perfectly with classification tasks, where the objective is to predict one of two classes.
# 
# 3. **Predictive Potential**:
#    - Other features in the dataset (e.g., `General_Health`, `BMI`, `Smoking_History`) are known risk factors for cardiovascular disease. These variables serve as predictors, while `Heart_Disease` is the outcome or dependent variable.
# 
# 4. **Actionable Insights**:
#    - Predicting `Heart_Disease` can provide actionable insights, identifying individuals at high risk and enabling early intervention or treatment.
# 
# 5. **Compatibility with Machine Learning Models**:
#    - Although currently stored as an object (`Yes/No`), `Heart_Disease` can be easily converted to a numerical format (e.g., `1` for `Yes` and `0` for `No`), making it compatible with machine learning algorithms.
# 
# By choosing `Heart_Disease` as the target variable, the project is focused on solving a meaningful healthcare problem and generating valuable predictive insights.
# 

# In[2]:


# Preview the data
print("\nFirst 5 rows of the dataset:")
print(df.head())


# #### Sample Data (First 5 Rows):
# | General_Health | Checkup                  | Exercise | Heart_Disease | Skin_Cancer | ... | BMI   | Smoking_History | Alcohol_Consumption | Fruit_Consumption | FriedPotato_Consumption |
# |----------------|--------------------------|----------|---------------|-------------|-----|-------|-----------------|---------------------|-------------------|-------------------------|
# | Poor           | Within the past 2 years | No       | No            | No          | ... | 14.54 | Yes             | 0.0                 | 30.0              | 12.0                    |
# | Very Good      | Within the past year    | No       | Yes           | No          | ... | 28.29 | No              | 0.0                 | 30.0              | 4.0                     |
# | Very Good      | Within the past year    | Yes      | No            | No          | ... | 33.47 | No              | 4.0                 | 12.0              | 16.0                    |
# | Poor           | Within the past year    | Yes      | Yes           | No          | ... | 28.73 | No              | 0.0                 | 30.0              | 8.0                     |
# | Good           | Within the past year    | No       | No            | No          | ... | 24.37 | Yes             | 0.0                 | 8.0               | 0.0                     |
# 
# This preview indicates the dataset contains both health-related factors (e.g., `General_Health`, `Exercise`) and demographic information (e.g., `Sex`, `Age_Category`), making it well-suited for predictive modeling. 
# The next steps will involve cleaning and preprocessing these variables to prepare the dataset for analysis.
# 

# In[3]:


# Dataset Info
print("\nDataset Info:")
print(df.info())

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Basic Visualization: Numeric Columns
numeric_columns = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption']
df[numeric_columns].hist(bins=15, figsize=(10, 8))
plt.tight_layout()
plt.show()

# Basic Visualization: Categorical Columns
categorical_columns = ['Sex', 'Smoking_History', 'General_Health']
for column in categorical_columns:
    sns.countplot(x=df[column])
    plt.title(f"Distribution of {column}")
    plt.show()


# ## Step 2: Data Preprocessing

# In[4]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

# Handle Missing Values
# For numeric columns, use mean imputation
numeric_columns = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 
                   'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

# For categorical columns, use most frequent imputation
categorical_columns = ['General_Health', 'Checkup', 'Exercise', 'Heart_Disease', 
                       'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 
                       'Arthritis', 'Sex', 'Smoking_History', 'Age_Category']
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# Feature Engineering: Add Age Midpoint from Age_Category
def calculate_age_midpoint(age_range):
    if age_range == "80+":  
        return 85
    else:
        lower, upper = map(int, age_range.split('-'))
        return (lower + upper) / 2

df['Age_Midpoint'] = df['Age_Category'].apply(calculate_age_midpoint)

# Encoding Categorical Variables
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  

# Save the rationale for feature engineering
feature_rationale = {
    "BMI": "Body Mass Index is a standard health metric.",
    "Age_Midpoint": "Approximation of age as a continuous variable from bins.",
    "Smoking_History": "Smoking history is a known risk factor for cardiovascular diseases.",
}

# Class Imbalance Check for Target Variable
target = "Heart_Disease"
class_counts = df[target].value_counts()
print(f"Class Distribution:\n{class_counts}\n")
print(f"Class Imbalance Ratio (Minority/Majority): {class_counts.min() / class_counts.max():.2f}")

# Optional: Visualize Class Distribution
sns.countplot(x=df[target])
plt.title("Target Class Distribution (Heart_Disease)")
plt.show()

# Check Preprocessed Data
print("Preview of Processed Data:")
print(df.head())


# ## Step 3: Exploratory Data Analysis (EDA)

# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

# Correlation Analysis
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[6]:


# Identify high correlations (absolute correlation > 0.7)
high_corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
high_corr_pairs = high_corr_pairs[(high_corr_pairs.abs() > 0.7) & (high_corr_pairs < 1)]
print("High Correlation Pairs:\n", high_corr_pairs)


# In[7]:


# Distribution Analysis for Numeric Features
numeric_columns = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
                   'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
for column in numeric_columns:
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.show()


# In[8]:


# Distribution Analysis for Target Variable
sns.countplot(x=df['Heart_Disease'])
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0: No, 1: Yes)")
plt.ylabel("Count")
plt.show()


# In[9]:


# Feature Importance (Optional Preview)
from sklearn.ensemble import RandomForestClassifier

# Define target and features
X = df.drop(columns=["Heart_Disease"])
y = df["Heart_Disease"]

# Fit a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance from Random Forest")
plt.show()



# In[10]:


# Outlier Detection for Numeric Features
from scipy.stats import zscore

for column in numeric_columns:
    z_scores = zscore(df[column])
    outliers = df[(z_scores > 3) | (z_scores < -3)]
    print(f"{column}: Found {len(outliers)} outliers.")


# In[11]:


# Visualize Outliers using Boxplots
for column in numeric_columns:
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot for {column}")
    plt.show()


# In[12]:


# Summary Statistics for Numeric Features
print("Summary Statistics for Numeric Features:")
print(df[numeric_columns].describe())


# ## Step 4: Data Splitting

# In[13]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Define target and features
target = "Heart_Disease"
features = df.drop(columns=[target])
labels = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

# Output the shape of splits
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# Check class balance in training and testing sets
print("Class distribution in training set:")
print(y_train.value_counts(normalize=True))

print("\nClass distribution in testing set:")
print(y_test.value_counts(normalize=True))

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Verify the new class distribution after SMOTE
print("\nClass distribution in training set after SMOTE:")
print(y_train_balanced.value_counts(normalize=True))


# ## Step 5: Feature Scaling

# In[14]:


from sklearn.preprocessing import StandardScaler
import joblib

# Identify numeric columns for scaling
numeric_columns = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
                   'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'Age_Midpoint']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the SMOTE-balanced training data and transform both training and testing data
X_train_balanced_scaled = X_train_balanced.copy()
X_test_scaled = X_test.copy()

X_train_balanced_scaled[numeric_columns] = scaler.fit_transform(X_train_balanced[numeric_columns])
X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Save the scaler for future use or deployment
joblib.dump(scaler, "cvd_predict_scaler.pkl")

# Verify scaling
print("Preview of scaled numeric features (training set after SMOTE):")
print(X_train_balanced_scaled[numeric_columns].head())


# ## Step 6: Model Building

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define Baseline Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Define Deep Learning Model
def build_deep_learning_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Get input dimension for deep learning model
input_dim = X_train_balanced_scaled.shape[1]  
deep_learning_model = build_deep_learning_model(input_dim)

# Summary of Deep Learning Model
deep_learning_model.summary()


# ## Step 7: Model Training

# In[16]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Define a function for evaluating the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):  
        y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    if y_proba is not None:
        print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Train baseline models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_balanced_scaled, y_train_balanced)  
    print(f"Evaluating {name}...")
    evaluate_model(model, X_test_scaled, y_test)  

# Train Deep Learning Model
print("\nTraining Deep Learning Model...")

# Callback for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Train the model
history = deep_learning_model.fit(
    X_train_balanced_scaled, y_train_balanced,  
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate Deep Learning Model
print("\nEvaluating Deep Learning Model...")
dl_y_pred = (deep_learning_model.predict(X_test_scaled) > 0.5).astype("int32").flatten()  
dl_y_proba = deep_learning_model.predict(X_test_scaled).flatten()

print("Accuracy:", accuracy_score(y_test, dl_y_pred))
print("Precision:", precision_score(y_test, dl_y_pred))
print("Recall:", recall_score(y_test, dl_y_pred))
print("F1 Score:", f1_score(y_test, dl_y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, dl_y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, dl_y_pred))

# Plot Training History for Deep Learning Model
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Deep Learning Model Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Deep Learning Model Loss')
plt.legend()
plt.show()


# ## Step 9: Model Optimization

# #### Optimized Logistic Regression (GridSearchCV)

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Parameter grid with reduced values
param_grid_lr = {
    "C": [0.1, 1, 10],
    "solver": ["liblinear"]
}

# Optimization
grid_search_lr = GridSearchCV(
    estimator=LogisticRegression(random_state=42, max_iter=500),
    param_grid=param_grid_lr,
    scoring="f1",
    cv=3,  
    verbose=1,
    n_jobs=-1  
)

# Fit and evaluate
grid_search_lr.fit(X_train_balanced_scaled, y_train_balanced)
best_lr = grid_search_lr.best_estimator_

print("\nBest parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best F1 Score for Logistic Regression:", grid_search_lr.best_score_)


# #### Optimized Random Forest (RandomizedSearchCV)

# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Optimized parameter grid
param_grid_rf = {
    "n_estimators": [50, 100],  
    "max_depth": [10, 20],      
    "min_samples_split": [2, 5],  
    "min_samples_leaf": [1, 2],   
    "class_weight": ['balanced']  
}

# Optimization with GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    scoring="f1",  
    cv=2,         
    verbose=1,
    n_jobs=-1      
)

# Subset data for faster fitting (optional)
subset_size = 10000  
X_subset = X_train_balanced_scaled[:subset_size]
y_subset = y_train_balanced[:subset_size]

# Fit and evaluate
grid_search_rf.fit(X_subset, y_subset)
best_rf = grid_search_rf.best_estimator_

# Display best parameters and F1 score
print("\nBest parameters for Random Forest:", grid_search_rf.best_params_)
print("Best F1 Score for Random Forest:", grid_search_rf.best_score_)

# Evaluate on the test set
y_pred = best_rf.predict(X_test_scaled)
y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

print("\nTest Set Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# #### Optimized Deep Learning with Keras Tuner

# In[30]:


import keras_tuner as kt
import tensorflow as tf

# Build model function for Keras Tuner
def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hp.Int('units_1', min_value=64, max_value=128, step=32), activation='relu'),
        tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.3, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(hp.Int('units_2', min_value=32, max_value=64, step=16), activation='relu'),
        tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.3, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Keras Tuner configuration
tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=10,  
    factor=3,  
    directory="hyperband_log",
    project_name="optimized_dl_tuning"
)

# Subset for tuning
subset_size = 20000
X_subset = X_train_balanced_scaled[:subset_size]
y_subset = y_train_balanced[:subset_size]

# Start search with early stopping
tuner.search(
    X_subset,
    y_subset,
    epochs=10,
    validation_split=0.2,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
)

# Get the best model
best_dl_model = tuner.get_best_models(num_models=1)[0]
print("\nBest hyperparameters for Deep Learning:")
tuner.results_summary()


# ## Retrain and Evaluate

# In[38]:


from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate dynamic class weights
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_balanced)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print("Updated Class Weights:", class_weights_dict)

# Define model architecture
best_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train model
history = best_model.fit(
    X_train_balanced_scaled, y_train_balanced,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = best_model.evaluate(X_test_scaled, y_test, verbose=1)
print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)



# In[56]:


# Save the trained model
best_model.save("best_model.h5")
print("Model saved as best_model.h5")


# In[39]:


# Classification report and confusion matrix
y_pred = (best_model.predict(X_test_scaled) > 0.5).astype("int32").flatten()
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[40]:


# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ## Model interpretability

# In[42]:


get_ipython().system('pip install shap')


# #### 1. SHAP Implementation

# In[43]:


import shap
import matplotlib.pyplot as plt

# Initialize SHAP explainer for deep learning model
explainer = shap.Explainer(best_model, X_test_scaled)

# Calculate SHAP values
shap_values = explainer(X_test_scaled)

# Global interpretation - SHAP Summary Plot
print("\nGenerating SHAP Summary Plot...")
shap.summary_plot(shap_values, X_test_scaled, feature_names=X_test_scaled.columns)

# Local interpretation - SHAP Force Plot for a Specific Prediction
idx = 0  
print(f"\nGenerating SHAP Force Plot for Prediction Index {idx}...")
shap.force_plot(explainer.expected_value[0], shap_values[idx].values, X_test_scaled.iloc[idx])


# #### 2. LIME Implementation

# In[45]:


get_ipython().system('pip install lime')


# In[48]:


from lime import lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# Initialize LIME explainer
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train_balanced_scaled.values,
    feature_names=X_train_balanced_scaled.columns,
    class_names=['No', 'Yes'],  
    mode='classification'
)

# Custom function for LIME to format predictions
def predict_proba_for_lime(data):
    preds = best_model.predict(data)
    # Convert single-class probabilities to two-class format: [P(class_0), P(class_1)]
    return np.hstack([(1 - preds), preds])

# Explain a single prediction
idx = 0  # Index of the prediction to explain
print(f"\nGenerating LIME Explanation for Prediction Index {idx}...")

# Generate the explanation
exp = explainer_lime.explain_instance(
    X_test_scaled.iloc[idx].values,
    predict_proba_for_lime,  
    num_features=10
)

# Display explanation
exp.show_in_notebook(show_table=True)
plt.figure(figsize=(8, 6))
exp.as_pyplot_figure()
plt.title(f'LIME Explanation for Prediction Index {idx}')
plt.show()


# #### 3. Feature Attribution (SHAP for Global Insights)

# In[49]:


# SHAP Feature Importance (Bar Plot)
print("\nGenerating SHAP Feature Importance Bar Plot...")
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", feature_names=X_test_scaled.columns)

# Beeswarm Plot for Visualizing Feature Attribution
print("\nGenerating SHAP Beeswarm Plot...")
shap.summary_plot(shap_values, X_test_scaled, feature_names=X_test_scaled.columns)


# #### 4. Domain Knowledge Validation
# Add these lines to validate if SHAP values align with domain knowledge:

# In[52]:


# Compare SHAP results with domain knowledge
print("\nValidating SHAP Results with Domain Knowledge...")

# Extract SHAP values as NumPy array
shap_values_array = shap_values.values  

# Compute mean absolute SHAP values for each feature
shap_values_mean = np.abs(shap_values_array).mean(axis=0)

# Pair feature names with their mean SHAP values and sort
sorted_features = sorted(zip(X_test_scaled.columns, shap_values_mean), key=lambda x: -x[1])

# Display top features influencing predictions
print("\nTop Features Influencing Predictions (By Mean Absolute SHAP Values):")
for feature, value in sorted_features[:10]: 
    print(f"{feature}: {value:.4f}")

# Example domain knowledge validation
print("\nDomain Knowledge Validation:")
if "Smoking_History" in [feature for feature, _ in sorted_features[:10]]:
    print("Smoking_History correctly identified as a top predictor.")
else:
    print("Smoking_History not in top predictors; investigate further.")


# #### 5. Enhancements
# Simulate changes to key features and observe the impact on predictions.

# In[53]:


# Simulate changes to Smoking_History feature and observe predictions
simulated_data = X_test_scaled.copy()
simulated_data["Smoking_History"] = 0  

# Get predictions on the simulated data
print("\nSimulating Non-Smoking Scenario and Observing Predictions...")
simulated_predictions = best_model.predict(simulated_data)
print("Average Predicted Risk (Non-Smoking Scenario):", simulated_predictions.mean())


# #### 6. Notes
# For SHAP, if the dataset is large, calculate SHAP values only for the top 100 rows using:

# In[54]:


shap_values = explainer(X_test_scaled.iloc[:100])


# ## 11. Integration, Reporting, and Deployment 

# #### 1. Complete End-to-End Pipeline
# Finalize preprocessing, model inference, and thresholding scripts:
# 
# Preprocessing & Model Inference Script

# In[62]:


import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Load scaler and model
scaler = joblib.load("cvd_predict_scaler.pkl")
model = load_model("best_model.h5")

# Define preprocessing and inference pipeline
def preprocess_and_predict(input_data):
   
    # Define all expected columns except the target column
    expected_columns = [
        'General_Health', 'Checkup', 'Exercise', 'Skin_Cancer',
        'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 'Sex',
        'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'BMI', 'Smoking_History',
        'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption',
        'FriedPotato_Consumption', 'Age_Midpoint'
    ]

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0  

    # Scale numeric columns
    numeric_columns = [
        'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
        'Fruit_Consumption', 'Green_Vegetables_Consumption',
        'FriedPotato_Consumption', 'Age_Midpoint'
    ]
    input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

    # Make predictions
    predictions = model.predict(input_data)
    predicted_classes = (predictions > 0.5).astype(int)

    return predicted_classes, predictions

# Example usage
if __name__ == "__main__":
    # Example input data
    data = {
        'Height_(cm)': [170],
        'Weight_(kg)': [70],
        'BMI': [24.2],
        'Alcohol_Consumption': [5],
        'Fruit_Consumption': [30],
        'Green_Vegetables_Consumption': [15],
        'FriedPotato_Consumption': [8],
        'Age_Midpoint': [40],
        'General_Health': [2],
        'Checkup': [0],
        'Exercise': [1],
        'Skin_Cancer': [0],
        'Other_Cancer': [0],
        'Depression': [0],
        'Diabetes': [0],
        'Arthritis': [0],
        'Sex': [1],
        'Age_Category': [4],
        'Smoking_History': [1]
    }
    input_df = pd.DataFrame(data)

    # Get predictions
    predicted_classes, probabilities = preprocess_and_predict(input_df)

    print("Predicted Classes:", predicted_classes.flatten())
    print("Probabilities:", probabilities.flatten())


# ## 2. Dashboard Creation (Streamlit)
# Create a dashboard for report generation:
# 
# Streamlit Dashboard Example

# In[ ]:


import streamlit as st 
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import numpy as np

# Load scaler and model
scaler = joblib.load("cvd_predict_scaler.pkl")
model = load_model("best_model.h5")

# Example true labels and predictions for metrics demonstration 
true_labels = [0, 1, 1, 0, 1]  
predicted_labels = [0, 1, 1, 0, 1]  

# Streamlit app
st.title("Heart Disease Prediction App")
st.markdown("Predict the likelihood of heart disease based on health and lifestyle factors.")

# Help Button
if st.button("Help"):
    st.markdown(
        """
        ### Contact for Support
        - **Name**: Md Zakir Hossain  
        - **Email**: zakir.minister@gmail.com  
        - **Phone**: +1 (715) 440-8235  

        *Feel free to reach out via email or phone if you have any questions, need assistance, or want to provide feedback. We aim to respond within 24 hours.*
        """
    )

# Input fields
height = st.slider("Height (cm)", 100, 200, 170)
weight = st.slider("Weight (kg)", 30, 150, 70)
bmi = st.slider("BMI", 10.0, 40.0, 24.2)
alcohol = st.slider("Alcohol Consumption (drinks/week)", 0, 30, 5)
fruit = st.slider("Fruit Consumption (servings/week)", 0, 30, 10)
veg = st.slider("Vegetable Consumption (servings/week)", 0, 30, 10)
potato = st.slider("Fried Potato Consumption (servings/week)", 0, 30, 5)
age = st.slider("Age", 18, 80, 40)
age_category = st.slider("Age Category (1=18-24, ..., 13=75+)", 1, 13, 4)
smoking = st.checkbox("Smoking History")
exercise = st.checkbox("Regular Exercise")
checkup = st.checkbox("Regular Medical Checkup")
health = st.selectbox("General Health (1=Excellent, 5=Poor)", [1, 2, 3, 4, 5])
sex = st.selectbox("Sex", ["Male", "Female"])
skin_cancer = st.checkbox("History of Skin Cancer")
other_cancer = st.checkbox("History of Other Cancer")
depression = st.checkbox("History of Depression")
diabetes = st.checkbox("History of Diabetes")
arthritis = st.checkbox("History of Arthritis")

# Process inputs
if st.button("Predict"):
    input_data = pd.DataFrame({
        "Height_(cm)": [height],
        "Weight_(kg)": [weight],
        "BMI": [bmi],
        "Alcohol_Consumption": [alcohol],
        "Fruit_Consumption": [fruit],
        "Green_Vegetables_Consumption": [veg],
        "FriedPotato_Consumption": [potato],
        "Age_Midpoint": [age],
        "Smoking_History": [1 if smoking else 0],
        "Exercise": [1 if exercise else 0],
        "Checkup": [1 if checkup else 0],
        "General_Health": [health],
        "Sex": [1 if sex == "Male" else 0],
        "Age_Category": [age_category],
        "Skin_Cancer": [1 if skin_cancer else 0],
        "Other_Cancer": [1 if other_cancer else 0],
        "Depression": [1 if depression else 0],
        "Diabetes": [1 if diabetes else 0],
        "Arthritis": [1 if arthritis else 0],
    })

    # Scale numeric features
    numeric_columns = [
        "Height_(cm)", "Weight_(kg)", "BMI", "Alcohol_Consumption",
        "Fruit_Consumption", "Green_Vegetables_Consumption",
        "FriedPotato_Consumption", "Age_Midpoint"
    ]
    input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

    # Predict
    probabilities = model.predict(input_data)
    predicted_class = int(probabilities[0] > 0.5)
    probability = float(probabilities[0])

    # Display Results
    if predicted_class == 1:
        result_text = f"High Risk of Heart Disease: {probability:.2%} confidence"
        st.error(result_text)
    else:
        result_text = f"Low Risk of Heart Disease: {1 - probability:.2%} confidence"
        st.success(result_text)

    # SHAP Explanation
    explainer = shap.Explainer(model, input_data)
    shap_values = explainer(input_data)

    st.subheader("Feature Importance Explanation")
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(plt)

    # Performance Metrics
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    metrics_report = classification_report(true_labels, predicted_labels, output_dict=True)

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    st.text(conf_matrix)

    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Heart Disease Prediction Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=result_text, ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Input Data:", ln=True)
    pdf.ln(10)
    for key, value in input_data.iloc[0].items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Confusion Matrix:", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=str(conf_matrix))
    pdf.ln(10)
    pdf.cell(200, 10, txt="Classification Report:", ln=True)
    pdf.ln(10)
    for key, value in metrics_report.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    # Provide download button for PDF
    st.download_button(
        label="Download Report as PDF",
        data=pdf_output,
        file_name="prediction_report.pdf",
        mime="application/pdf",
    )

    # Generate Excel Report
    excel_output = io.BytesIO()
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        input_data.to_excel(writer, sheet_name="Inputs", index=False)
        pd.DataFrame({"Prediction": [result_text]}).to_excel(writer, sheet_name="Prediction", index=False)
        pd.DataFrame(conf_matrix).to_excel(writer, sheet_name="Confusion Matrix", index=False)
        pd.DataFrame(metrics_report).to_excel(writer, sheet_name="Metrics Report")
    excel_output.seek(0)

    # Provide download button for Excel
    st.download_button(
        label="Download Report as Excel",
        data=excel_output,
        file_name="prediction_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )



# In[ ]:




