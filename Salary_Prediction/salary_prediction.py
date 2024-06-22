# Import necessary libraries
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

# Load dataset
data = pd.read_csv('Salary Prediction of Data Professions.csv')

# Display first few rows of the dataset
data.head()

# 1. Exploratory Data Analysis (EDA)

# Basic statistics
data.describe()

# Check for missing values
data.isnull().sum()

# Distribution of the target variable (salary)
plt.figure(figsize=(10, 6))
sns.histplot(data['SALARY'], kde=True)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# Visualize relationships between numerical features and the target variable
numerical_features = ['AGE', 'LEAVES USED',
                      'LEAVES REMAINING', 'RATINGS', 'PAST EXP']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=feature, y='SALARY')
    plt.title(f'Salary vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Salary')
    plt.show()

# Count plot for categorical variables
categorical_features = ['SEX', 'DESIGNATION', 'UNIT']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=feature)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# 2. Feature Engineering
# Creating a new feature TENURE based on DOJ and CURRENT DATE, and handle categorical variables appropriately.

# Convert date columns to datetime
data['DOJ'] = pd.to_datetime(data['DOJ'])
data['CURRENT DATE'] = pd.to_datetime(data['CURRENT DATE'])

# Create a new feature 'TENURE'
data['TENURE'] = (data['CURRENT DATE'] - data['DOJ']).dt.days / 365.25

# Drop 'DOJ' and 'CURRENT DATE' as they are no longer needed
data.drop(['DOJ', 'CURRENT DATE'], axis=1, inplace=True)

# Ensure categorical features are treated as strings
data['SEX'] = data['SEX'].astype(str)
data['DESIGNATION'] = data['DESIGNATION'].astype(str)
data['UNIT'] = data['UNIT'].astype(str)

# 3. Data Preprocessing
# Handle missing values and scale the features.

# Define the features and target variable
X = data.drop(columns=['FIRST NAME', 'LAST NAME', 'SALARY'])
y = data['SALARY']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines for numerical and categorical features
numerical_features = ['AGE', 'LEAVES USED',
                      'LEAVES REMAINING', 'RATINGS', 'PAST EXP', 'TENURE']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['SEX', 'DESIGNATION', 'UNIT']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the full pipeline with preprocessing and model


def create_pipeline(model):
    return Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# 4. Machine Learning Model Development
# Experimenting with different models.


# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

# 5. Model Evaluation
# We'll summarize the evaluation metrics and identify the best-performing model.

# Display results
results_df = pd.DataFrame(results).T
print(results_df)

# 6. ML Pipelines and Model Deployment
# We'll create an ML pipeline for the best-performing model and prepare it for deployment using Flask.

# Assume RandomForest is the best model based on evaluation
best_model = RandomForestRegressor(random_state=42)
best_pipeline = create_pipeline(best_model)
best_pipeline.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(best_pipeline, 'salary_prediction_model.pkl')

# Flask app for model deployment

app = Flask(__name__)

# Load the trained model
model = joblib.load('salary_prediction_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'salary': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
