from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pylab as plt

import dmba
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold,train_test_split, cross_val_score, GridSearchCV, KFold #GridSearch is for hyperparameter tuning
from sklearn.preprocessing import StandardScaler,OneHotEncoder, MinMaxScaler, LabelEncoder, FunctionTransformer, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score, precision_score, recall_score, PrecisionRecallDisplay, RocCurveDisplay, make_scorer, mean_squared_error,classification_report,confusion_matrix,ConfusionMatrixDisplay, roc_curve, auc,roc_auc_score
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier


# EDA

df = pd.read_csv('')

df.head()

df.info()

df.groupby('customer_id')['transaction_amount'].transform('sum')

df.describe()

sns.boxplot()

# check for number of duplicates
df.duplicated().sum()

# visualize distribution of 'is_canceled'
sns.countplot(x='is_canceled', data=df, hue='is_canceled', legend=False)
plt.title('Distribution of Cancellations')
plt.show()

# visualization of the top 10 countries that cancelled bookings are from
top_country_cancellations = country_cancellations.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_country_cancellations.index, y=top_country_cancellations.values, hue=top_country_cancellations, legend=False)
plt.title('Top 10 Countries with Most Cancellations')
plt.xlabel('Country')
plt.ylabel('Number of Cancellations')
plt.show()

# heatmap of correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# visualize cancellations by month
plt.figure(figsize=(14, 7))
sns.barplot(x=monthly_cancellations.index, y=monthly_cancellations.values, hue=monthly_cancellations.index, legend=False)
plt.title('Monthly Cancellations')
plt.xlabel('Month')
plt.ylabel('Number of Cancellations')
plt.xticks(rotation=45)
plt.show()

# FEATURE ENGINEERING

# Convert 'Yes' to 1 and 'No' to 0
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})



# Calculate the 99th percentile of the 'transaction_amount' column
percentile_99 = df['LTV'].quantile(0.99)



# Replace values above the 99th percentile with the 99th percentile value
df['LTV'] = df['LTV'].apply(lambda x: min(x, percentile_99))



# Define the age bins and labels
bins = [18, 35, 50, 65, float('inf')]
labels = ['18-34', '35-49', '50-64', '65+']



# Create the age bins and label them
df['age_bins'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

df['age_bins'].value_counts()



# Create column to track customer_lifespan
df['customer_lifespan'] = df['last_purchase_date'] - df['join_date']


# We want to predict ranges of LTV for better use in model output

# Define the LTV bins and labels(25%,50%,75%,infinity)
ltv_bins = [0, 7134.53, 10324.62, float('inf')]
ltv_labels = ['Low', 'Medium', 'High']

# Create the LTV_cat column in the transaction_level DataFrame
df['LTV_cat'] = pd.cut(df['LTV'], bins=ltv_bins, labels=ltv_labels, right=False)


# Group by 'Job Role' and calculate the sum and count of 'Attrition'
summary = df.groupby('ColumnCheck')['Output'].agg(['sum', 'count'])

# Calculate the percentage of attrition (1s) in each job role
summary['Output_Percentage'] = (summary['sum'] / summary['count']) * 100

# Display the result
print(summary[['Output_Percentage']])







# MODELLING

# Split Data for Model

X = df[['age_bins','gender','location','number_of_site_visits','number_of_emails_opened','number_of_clicks','transaction_amount','customer_lifespan']]
y = df['LTV_cat']

numeric_columns = ['number_of_site_visits','number_of_emails_opened','number_of_clicks','transaction_amount','customer_lifespan']
categorical_columns = ['age_bins','gender','location']

# reserve 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# create a pre-processing pipeline which includes the steps of Scaling numeric variables and encoding categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ('num',MinMaxScaler(), numeric_columns),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical_columns)
    ]
)

# List of models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Evaluate each model using cross-validation
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, scoring='f1_weighted', cv=3,n_jobs=-1)
    print(f"{name} F1 Scores (weighted): {scores.mean():.3f} ± {scores.std():.3f}")



# Binary Log Reg Pipeline

# Define the Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logreg)
])

# Fit the model to the training data
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))


# Logreg Pipeline - Multi-Class

logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

logreg_pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = logreg_pipeline.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred) 
print("Classification Report:\n", report)

# Compute and print individual metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# KNN

# KNN
KNN = KNeighborsClassifier()

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNN)
])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fit the model to the training data
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Random Forest

# Define the preprocessor with OneHotEncoder for categorical features and MinMaxScaler for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a pipeline with the preprocessor and a RandomForestClassifier
pipeline_rforest_rev = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train the model
pipeline_rforest_rev.fit(X_train, y_train)

# Make predictions
y_pred = pipeline_rforest_rev.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Only for continuous variables

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Output the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


# Troubleshooting Columns Missing

# Check if all specified columns are present in the DataFrame
missing_columns = set(numeric_columns + categorical_columns) - set(X_train.columns)

if missing_columns:
    print(f"Missing columns in the DataFrame: {missing_columns}")
else:
    print("All columns are present.")

 # Check data types of the columns in the DataFrame
print(X_train.dtypes)

# Check for duplicate columns
duplicate_columns = X_train.columns[X_train.columns.duplicated()]
if not duplicate_columns.empty:
    print(f"Duplicate columns found: {duplicate_columns}")
else:
    print("No duplicate columns found.")

# Check for unique values in categorical columns to ensure encoding will work correctly
for column in categorical_columns:
    unique_values = X_train[column].unique()
    print(f"Column: {column}, Unique Values: {unique_values}")




# Grid Search Params for All Models:

param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['lbfgs', 'liblinear']
    },
    'KNN': {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']  # Common distance metrics
    },
    'Naive Bayes': {
        # GaussianNB doesn't have hyperparameters typically tuned via GridSearch
    },
    'Decision Tree': {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30]
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    }
}


# GridSearch Run (Swap Models & Params)

def run_grid_search(pipeline, param_grid, X_train, y_train):
    """
    Run GridSearchCV on the provided pipeline and parameter grid.

    Parameters:
    - pipeline: sklearn.pipeline.Pipeline object with preprocessor and model
    - param_grid: dict, parameter grid for GridSearchCV
    - X_train: Training features
    - y_train: Training labels

    Returns:
    - best_estimator: Best model from GridSearchCV
    - best_params: Best parameters from GridSearchCV
    - best_score: Best score achieved by the best model
    """
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    return best_estimator, best_params, best_score

# Assuming 'preprocessor' is defined and 'X_train', 'y_train' are your training data

# Replace with your selected model and corresponding parameter grid
selected_model = RandomForestClassifier() # Put model you want to use here
param_grid = {
    'classifier__n_estimators': [100, 200, 300], # Change Params using above listed arrays
    'classifier__max_depth': [None, 10, 20, 30]
}

# Create a pipeline with the selected model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', selected_model)
])

# Run grid search
best_estimator, best_params, best_score = run_grid_search(pipeline, param_grid, X_train, y_train)

print("Best Estimator:", best_estimator) # Output gives Model with Parameters assigned
print("Best Parameters:", best_params)
print("Best Score:", best_score)