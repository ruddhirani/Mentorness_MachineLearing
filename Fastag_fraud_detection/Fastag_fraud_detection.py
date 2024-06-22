import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('FastagFraudDetection.csv')

# Data Exploration
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Distribution of fraud indicator
sns.countplot(x='Fraud_indicator', data=df)
plt.title('Distribution of Fraud Indicator')
plt.show()

# Data Preprocessing and Feature Engineering
# Convert Timestamp to datetime and extract useful time-based features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(
    df, columns=['Vehicle_Type', 'Lane_Type', 'Geographical_Location'])

# Feature: Transaction Amount Difference
df['Amount_Difference'] = df['Transaction_Amount'] - df['Amount_paid']

# Drop unnecessary columns
df = df.drop(columns=['Transaction_ID', 'Timestamp', 'Vehicle_Plate_Number',
             'FastagID', 'TollBoothID', 'Vehicle_Dimensions'])

# Check the dataframe after feature engineering
print(df.head())

# Model Development
# Split the data into features and target variable
X = df.drop(columns=['Fraud_indicator'])
y = df['Fraud_indicator']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Model Evaluation
# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Real-time Fraud Detection Feasibility
# Serialize the model
joblib.dump(model, 'fastag_fraud_detection_model.pkl')

# Load the model and make a prediction (simulation of real-time prediction)
loaded_model = joblib.load('fastag_fraud_detection_model.pkl')
sample_transaction = X_test[0].reshape(1, -1)
fraud_prediction = loaded_model.predict(sample_transaction)

print(f'Fraud prediction for sample transaction: {fraud_prediction[0]}')

# Explanatory Analysis
# Feature importance
feature_importances = model.feature_importances_
features = X.columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame(
    {'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()
