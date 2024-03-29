# import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load data
def load_data():
    # read csv
    df = pd.read_csv("stroke prediction\healthcare-dataset-stroke-data.csv")
    # drop 'id' column, not needed for analysis/modeling
    df = df.drop('id', axis=1)
    
    # split columns into categorical and numerical
    categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numerical = ['age', 'avg_glucose_level', 'bmi']
    
    # separates the dataset into features (X) and target variable (y)
    y = df['stroke']
    X = df.drop('stroke', axis=1)

    return df, categorical, numerical, X, y

# Preprocessing and modeling
def random_forest_model(X,y):
    # Define preprocessing steps for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # handle missing values by replacing with medican
        ('scaler', StandardScaler()) # standardize the features
    ])

    # Define preprocessing steps for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # handle missing values by replacing with a constant ('missing')
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # encode categorical variables
    ])

    # Combine preprocessing steps for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical), # apply numeric_transformer to features in numerical list
            ('cat', categorical_transformer, categorical) # apply categorical_transformer to features in categorical list
        ])
    
     # Define the pipeline for modeling
    pipeline_rf = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE()),  # Handling class imbalance
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define parameter grids for hyperparameter tuning
    param_grid_rf = {
        'classifier__n_estimators': [100, 200, 300], # number of trees
        'classifier__max_depth': [5, 10, 15], # controls the max. depth of each tree
        'classifier__min_samples_split': [2, 5, 10] # determine the min. number of samples required to split an internal node
    }

    # Perform GridSearchCV for Random Forest
    grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='roc_auc') # searches over the parameter grid to find the best combination of hyperparameters for the classifier
    grid_search_rf.fit(X, y) # fit the grid search to the data
    best_rf_model = grid_search_rf.best_estimator_ #encapsulate the best combination of preprocessing steps and classifier

    return best_rf_model

def xgboost_model(X,y):
    # define preprocessing steps for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # imputes missing values using the median strategy
        ('scaler', StandardScaler()) # scale the featues using standardization (z-score normalization)
    ])

    # define preprocessing steps for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # imputes missing values with a constant fill value ('missing')
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # perform one-hot encoding to convert variables into numerical format
    ])

    # Combine preprocessing steps to numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical), #apply the preprocessing steps to the numerical & categorical features separately
            ('cat', categorical_transformer, categorical)
        ])
    
    # imbalanced pipeline
    pipeline_xgb = ImbPipeline(steps=[
        ('preprocessor', preprocessor), #preprocessing
        ('smote', SMOTE()),  # Handling class imbalance
        ('classifier', XGBClassifier(random_state=42)) #XGBoost classifier
    ])

    # parameter grid for hyperparameter tuning
    param_grid_xgb = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }

    # Perform GridSearchCV for XGBoost
    grid_search_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=5, scoring='roc_auc')
    grid_search_xgb.fit(X, y) #fit the grid search to the data
    best_xgb_model = grid_search_xgb.best_estimator_

    return best_xgb_model

# Evaluate models
def evaluate_models(models, X_test, y_test):
    # iterates over each key-value pair in models dict
    for name, model in models.items():
        # make predictions
        y_pred = model.predict(X_test)
        # print classification report
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))

        # print confusion matrix -> summarize the performance of a classification algorithm by tabulating the number of true positives, false positives, true negatives, and false negatives
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # print accuracy
        print("Accuracy:", accuracy_score(y_test, y_pred))
        
        # print ROC AUC score
        print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
        
        # print Precision, Recall, and F1 Score
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))
        print("\n")

def calculate_composite_score(metrics):
    # Define weights for each metric
    weights = {'accuracy': 1, 'precision': 1, 'recall': 1, 'f1_score': 1}

    # Calculate composite score by summing up the product of each metric's value and its corresponding weight
    composite_score = sum(weights[metric] * value for metric, value in metrics.items())

    return composite_score

def select_best_model(models, X_test, y_test):
    # initialize variavles
    best_model = None
    best_score = -1  # Initialize with a value lower than any possible score

    # iterate over each model
    for name, model in models.items():
        # make prediction
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        # calculate composite score
        composite_score = calculate_composite_score(metrics)
        # print composite score
        print(f"Composite Score for {name}: {composite_score}")
        
        # update the best model with its composite score
        if composite_score > best_score:
            best_model = model
            best_score = composite_score

    return best_model

# main method
if __name__ == "__main__":
    # Load data
    df, categorical, numerical, X, y = load_data()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data and train models using the training data
    rf_model = random_forest_model(X,y)
    xgb_model = xgboost_model(X, y)

    models = {"Random Forest": rf_model, "XGBoost": xgb_model}
    # evaluate models using the test data
    evaluate_models(models, X_test, y_test)
    # select the best model
    best_model = select_best_model(models, X_test, y_test)
    print(f"The best model is: {best_model.named_steps['classifier']}")

    # create sample data
    new_data = pd.DataFrame({
        'gender': ['Female', 'Male', 'Male'],  # Categorical feature
        'age': [55, 65, 40],  # Numerical feature
        'hypertension': [1, 0, 0],  # Binary feature
        'heart_disease': [0, 1, 0],  # Binary feature
        'ever_married': ['Yes', 'Yes', 'No'],  # Categorical feature
        'work_type': ['Private', 'Self-employed', 'Private'],  # Categorical feature
        'Residence_type': ['Urban', 'Rural', 'Urban'],  # Categorical feature
        'avg_glucose_level': [85.5, 120.3, 95.7],  # Numerical feature
        'bmi': [30.2, 26.8, 22.5],  # Numerical feature
        'smoking_status': ['never smoked', 'formerly smoked', 'never smoked'],  # Categorical feature
        'stroke': [0, 1, 0]  # Target variable
    })

    # Display sample data
    print(new_data) 

    # Make predictions on sample data using the best model
    new_data_X = new_data.drop('stroke', axis=1)  # Extract features for prediction
    new_data_y_pred = best_model.predict(new_data_X)  # Make predictions

    # Display the predictions on sample data
    print("Predictions on new data:")
    print(new_data_y_pred)

