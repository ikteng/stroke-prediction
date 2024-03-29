# Stroke Prediction

Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

This is an end-to-end machine learning project for predicting stroke occurences based on a healthcare dataset. 

I highly suggest checking out: https://www.kaggle.com/code/kaanboke/beginner-friendly-end-to-end-ml-project-enjoy/notebook

## stroke_prediction.py
Here is the breakdown of the main components of the code:

### Importing Libraries
- pandas
- numpy
- matplotlib
- sklearn
- xgboost
- imlearn

### Load Data
- loads dataset from a CSV file
- drops the 'id' column as it is not needed for analysis or modeling
- splits the columns into categorical and numerical features
- separate the dataset into features (X) and the target variable (y)

### Data Exploration
- explore the dataset's structure, including its dimension and data types
- identified that the dataset contains both numerical and categorical features, with binary classification for stroke prediction
- note that the dataset suffers from class imbalance, with only a small percentage of patients having a stroke

#### Missing Values
- detected a missing values in the 'bmi' column and planned to handle them using imputation within a pipeline

#### Univariate analysis
- conducted univariate analysis for numerical and ategorical features, including historgrams and bar plots to visualize distribution and frequencies respectively
- noted some trends such as the proportion of stroke cases among different categories (e.g., gender, hypertension, heart disease, etc.)

#### Bivariate Analysis
- explored relationships between features and the target variable (stroke), revealing insights such as the impact of hypertension, heart disease, marital status, work type, etc., on the likelihood of having a stroke.

#### Correlation Analysis
- calculated correlation coefficients between numerical features, identifying weak positive correlations between some features.
- visualized scatter plots to explore relationships between pairs of numerical features and their connection to stroke incidence.

#### Insights from Exploratory Data Analysis:
- Age and target variable weak positive relationship (almost .25).
- Average glucose level's mean scores on the target have small differences between a person who has a stroke or not.
- BMI does not have any significant relationship with the target variable.
- Male compare to female are more likelyto get stroke, but difference between female and male is very small.
- Person who lives in rural area has slightly more probablity to get stroke than a person who lives in rural area.
- It is small difference between who smokes and who does not smoke in regard to probability of getting stroke.

- A person with hypertension are almost 3.3 time more likely to get stroke than the ones who don't have hypertension.
- A person with heart diease are 4.07 times more likely to get stroke than the ones who don't have heart disease.
- A person is married(or married before) are 5.7 times more likely to get stroke than the ones who don't have marriage history.
- Self employed person has more probability to get stroke than other work type.

### Preprocessing and Modeling
- implement functions for building Random Forest and XGBoost models
- each function defines preprocessing steps for numerical and categorical features using pipelines and transformers
- the pipelines include imputation of missing values and one-hot encoding for categorical variables
- the models are trained using the training data and hyperparameters are tuned using GridSearchCV

### Evaluate Models
- evalute the trained models using test data
- metrics such as classification report, confusion matrix, accuracy, ROC AUC score, precision, recall, and F1 score are calculated and printed for each model

### Composite Score Calculation
- calculate a composite score based on multiple evaluation metrics
- each metric is assigned a weight, and the composite score is calculated as the sum of the products of each metric's value and its weight

### Select Best Model
- select the best performing model based on the composite score
- iterates over each model, makes predictions, calculates evaluation metrics, and computes the composite score
- the model with the highest composite score is selected as the best model

### Main Method
- loads the data, splits it into train and test sets, and trains the Random Forest and XGBoost models
- the trained models are evaluated using the test data, and the best model is selected.
- the best model is used to make predictions on sample data, and the predictions are printed.

#### Why does the model uses X, y instead of X_train, y_train?
- Cross-Validation:
  - The GridSearchCV function is used within these model functions to perform hyperparameter tuning using cross-validation.
  - When calling fit(X, y) on GridSearchCV, it internally performs cross-validation using the provided data (X and y) to evaluate different hyperparameter combinations.
  - Therefore, it's common practice to use the entire dataset (X and y) for this purpose.

- Imbalanced Data Handling:
  - Both functions include steps to handle imbalanced data using SMOTE (imblearn.over_sampling.SMOTE).
  - These steps require knowledge of the entire dataset to properly balance the classes.
  - Thus, it's necessary to use the complete dataset (X and y) for these preprocessing steps.

- Preprocessing:
    - The preprocessing steps, such as imputation of missing values and encoding categorical features, are applied to the entire dataset.
    - These steps need to be performed on the full dataset to ensure consistency and accuracy in handling missing values and encoding categorical features.

- Best Model Selection:
  - After hyperparameter tuning and model fitting, the select_best_model function evaluates the models using the test dataset (X_test and y_test).
  - This function selects the best-performing model based on the evaluation metrics computed on the test set.
