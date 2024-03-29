# import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, mutual_info_score
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
    df = pd.read_csv(r"stroke prediction\healthcare-dataset-stroke-data.csv")
    # drop 'id' column, not needed for analysis/modeling
    df = df.drop('id', axis=1)
    
    # split columns into categorical and numerical
    categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numerical = ['age', 'avg_glucose_level', 'bmi']
    
    # separates the dataset into features (X) and target variable (y)
    y = df['stroke']
    X = df.drop('stroke', axis=1)

    return df, categorical, numerical, X, y

# exploratory data analysis
def data_anaysis(df, categorical, numerical):
    # see the top 5 instances of the data
    print(df.head())

    print("Number of Entries: ", df.shape[0])
    print("Number of features: ", df.shape[1]-1)

    # calculate the number of duplicate rows in df
    df.duplicated().sum()

    # print the summary of df
    df.info()

    """
    Things to notice from info:
    - stroke is an int, not as an object
        -> target variable
        -> 1 for positive cases (has stroke), 0 for negative cases (no stroke)
    - hypertension and heart disease are detected as an int, not as an object
        -> 1 for positve cases (has hypertesion/heart disease), 0 for negative cases (no hypertension/heart disease)
    - 3 categorical variables, need to encode as numerical
    - binary classification problem
    """

    y = df['stroke']
    print(f'Percentage of patient had a stroke: % {round(y.value_counts(normalize=True)[1]*100,2)} --> ({y.value_counts()[1]} patient)')
    print(f'Percentage of patient did not have a stroke: % {round(y.value_counts(normalize=True)[0]*100,2)} --> ({y.value_counts()[0]} patient)')

    # count occurrences of each value in the 'stroke' column
    stroke_counts=y.value_counts()

    # create a bar plot
    plt.bar(stroke_counts.index, stroke_counts.values)

    # Set labels and title
    plt.xlabel('Stroke')
    plt.ylabel('Count')
    plt.title('Stroke')

    # Show plot
    plt.show()

    """
    From the stroke variable, 
    ~5% (249 patients) have stroke, ~95% (4861 patients) do not have stroke
    -> imbalanced data!
    """

    # dealing missing values
    def missing (df):
        missing_number = df.isnull().sum().sort_values(ascending=False)
        missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
        missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
        return missing_values

    print(missing(df)) # missing values on 'bmi', handle with pipeline

    # numerical features
    print(df[numerical].describe())

    # skewness
    print(df[numerical].skew())

    # univariate analysis
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(numerical):
        plt.subplot(3, 1, i+1)
        plt.hist(df[feature], bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # categorical features
    print(f'{round(df["gender"].value_counts(normalize=True)*100,2)}')
    
    gender_counts = df['gender'].value_counts()

    # Plotting the histogram
    plt.figure(figsize=(6, 6))
    plt.bar(gender_counts.index, gender_counts.values, color='skyblue', edgecolor='black')
    plt.title('Gender')
    plt.xlabel('Gender')
    plt.ylabel('Frequency')
    plt.show()
    # We have 2994 female and 2115 male and 1 other gender people.

    #hypertension
    print (f'{round(df["hypertension"].value_counts(normalize=True)*100,2)}')

    hypertension_counts = df['hypertension'].value_counts()

    # Plotting the histogram
    plt.figure(figsize=(6, 6))
    plt.bar(hypertension_counts.index.astype(str), hypertension_counts.values, color='skyblue', edgecolor='black')
    plt.title('Hypertension')
    plt.xlabel('Hypertension')
    plt.ylabel('Frequency')
    plt.show()
    #We have 498 patient with hypertension which represents at raound 10 % of the sample.

    # heart disease
    print (f'{round(df["heart_disease"].value_counts(normalize=True)*100,2)}')

    heart_disease_counts = df['heart_disease'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.bar(heart_disease_counts.index.astype(str), heart_disease_counts.values, color='skyblue', edgecolor='black')
    plt.title('Heart Disease')
    plt.xlabel('Heart Disease')
    plt.ylabel('Frequency')
    plt.show()
    #We have 276 patient with heart disease which is 5.4 % of the sample.

    print (f'{round(df["ever_married"].value_counts(normalize=True)*100,2)}')

    ever_married_counts = df['ever_married'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.bar(ever_married_counts.index.astype(str), ever_married_counts.values, color='skyblue', edgecolor='black')
    plt.title('Ever Married')
    plt.xlabel('Ever Married')
    plt.ylabel('Frequency')
    plt.show()
    #3353 people have been married and 1757 people are not married before.

    #work type
    print (f'{round(df["work_type"].value_counts(normalize=True)*100,2)}')

    work_type_counts = df['work_type'].value_counts()

    plt.figure(figsize=(8, 6))
    plt.bar(work_type_counts.index, work_type_counts.values, color='skyblue', edgecolor='black')
    plt.title('Work Type')
    plt.xlabel('Work Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    #2925 people work in the private sector; 819 people are self-employed; 657 people work at the government job.

    # residence type
    print (f'{round(df["Residence_type"].value_counts(normalize=True)*100,2)}')

    residence_type_counts = df['Residence_type'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.bar(residence_type_counts.index, residence_type_counts.values, color='skyblue', edgecolor='black')
    plt.title('Residence Type')
    plt.xlabel('Residence Type')
    plt.ylabel('Frequency')
    plt.show()
    # 2596 people live in the urban area; 2514 people live in the rural area

    # smoking
    print (f'{round(df["smoking_status"].value_counts(normalize=True)*100,2)}')

    smoking_status_counts = df['smoking_status'].value_counts()

    plt.figure(figsize=(8, 6))
    plt.bar(smoking_status_counts.index, smoking_status_counts.values, color='skyblue', edgecolor='black')
    plt.title('Smoking Status')
    plt.xlabel('Smoking Status')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
    # 1892 people are never smoked; 789 people smoke

    # Bivariate Analysis

    # hypertension & stroke
    print (f'A person with hypertension has a probability of {round(df[df["hypertension"]==1]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A person without hypertension has a probability of  {round(df[df["hypertension"]==0]["stroke"].mean()*100,2)} % get a stroke')

    # gender & stroke
    print (f'A female person has a probability of {round(df[df["gender"]=="Female"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A male person has a probability of {round(df[df["gender"]=="Male"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A person from  the other category of gender has a probability of {round(df[df["gender"]=="Other"]["stroke"].mean()*100,2)} % get a stroke')

    # heart disease & stroke
    print (f'A person with heart disease has a probability of {round(df[df["heart_disease"]==1]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A person without heart disease has a probability of {round(df[df["heart_disease"]==0]["stroke"].mean()*100,2)} % get a stroke')

    # marrid & stroke
    print (f'A person married (or married before) has a probability of {round(df[df["ever_married"]=="Yes"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A person never married has a probability of {round(df[df["ever_married"]=="No"]["stroke"].mean()*100,2)} % get a stroke')

    # work type & stroke
    print (f'A person with private work type has a probability of {round(df[df["work_type"]=="Private"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'Self-employed person has a probability of {round(df[df["work_type"]=="Self-employed"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A person with a goverment job has a probability of {round(df[df["work_type"]=="Govt_job"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A child has a probability of {round(df[df["work_type"]=="children"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A person never worked has a probability of {round(df[df["work_type"]=="Never_worked"]["stroke"].mean()*100,2)} % get a stroke')

    # residence type & stroke
    print (f'A person, who lives in urban area, has a probability of {round(df[df["Residence_type"]=="Urban"]["stroke"].mean()*100,2)} %  get a stroke')
    print()
    print (f'A person, who lives in rural area, has a probability of {round(df[df["Residence_type"]=="Rural"]["stroke"].mean()*100,2)} % get a stroke')

    # smoking & stroke
    print (f'A formerly smoked person has a probability of {round(df[df["smoking_status"]=="formerly smoked"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A person never smoked has a probability of {round(df[df["smoking_status"]=="never smoked"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A person smokes has a probability of {round(df[df["smoking_status"]=="smokes"]["stroke"].mean()*100,2)} % get a stroke')
    print()
    print (f'A person whom smoking history is not known,has a probability of {round(df[df["smoking_status"]=="Unknown"]["stroke"].mean()*100,2)} % get a stroke')
    print()

    # metrics of importance
    def cat_mut_inf(series):
        return mutual_info_score(series, df['stroke']) 

    df_cat = df[categorical].apply(cat_mut_inf) 
    df_cat = df_cat.sort_values(ascending=False).to_frame(name='mutual_info_score') 
    print(df_cat) # no effect

    # correlation matrix
    print(df[numerical].corr()) # very small positive correlation between numerical features

    print(df.groupby('stroke')[numerical].mean())

    print(df[['age','avg_glucose_level','bmi','stroke']].corr())

    # scatter plots

    # age & BMI
    plt.figure(figsize=(10, 6))
    plt.scatter(df['age'], df['bmi'], c=df['stroke'], cmap='viridis')
    plt.title('Age & BMI')
    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.colorbar(label='Stroke')
    plt.grid(True)
    plt.show()

    # age & average glucose level
    plt.figure(figsize=(10, 6))
    plt.scatter(df['age'], df['avg_glucose_level'], c=df['stroke'], cmap='viridis')
    plt.title('Age & Average Glucose Level')
    plt.xlabel('Age')
    plt.ylabel('Average Glucose Level')
    plt.colorbar(label='Stroke')
    plt.grid(True)
    plt.show()

    # average glucose level & BMI
    plt.figure(figsize=(10, 6))
    plt.scatter(df['bmi'], df['avg_glucose_level'], c=df['stroke'], cmap='viridis')
    plt.title('Average Glucose Level & BMI')
    plt.xlabel('BMI')
    plt.ylabel('Average Glucose Level')
    plt.colorbar(label='Stroke')
    plt.grid(True)
    plt.show()

    """
    Insights from Exploratory Data Analysis:
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
 
    """

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

    data_anaysis(df, categorical, numerical)

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

