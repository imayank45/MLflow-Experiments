from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

# load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# random forest classifier
rf = RandomForestClassifier(random_state=42)

# hyperparameter for gridsearchcv

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# apply gridsearchcv function
grid_search_cv = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1
)

# # fit the model
# grid_search_cv.fit(X_train, y_train)

# # get best parameters and best score

# best_params = grid_search_cv.best_params_
# best_score = grid_search_cv.best_score_

# # print best_score and best_params

# print("Best Parameters: ", best_params)
# print("Best Score: ", best_score)

# set experiment
mlflow.set_experiment("Breast-Cancer-Random-Forest-Classifier")

with mlflow.start_run() as parent:  
    
    grid_search_cv.fit(X_train, y_train)
    
    # log all child runs
    for i in range(len(grid_search_cv.cv_results_['params'])):
        
        with mlflow.start_run(nested = True):
            
            # log parameters and metrics
            mlflow.log_params(grid_search_cv.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search_cv.cv_results_['mean_test_score'][i])
    
    # get best parameters and best score
    best_params = grid_search_cv.best_params_
    best_score = grid_search_cv.best_score_
    
    # log parameters
    mlflow.log_params(best_params)
    
    # log metrics
    mlflow.log_metric("best_score", best_score)
    
    # log training data
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")
    
    # log testing data
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")
    
    # log source code
    mlflow.log_artifact(__file__)
    
    # log the best model
    mlflow.sklearn.log_model(grid_search_cv.best_estimator_, "best_model")
    
    # set tags
    mlflow.set_tag("author", "Mayank Kathane")
    mlflow.set_tag("algorithm", "Random Forest")
    mlflow.set_tag("dataset", "Breast Cancer")
    mlflow.set_tag("parameters", str(best_params))
    mlflow.set_tag("accuracy", best_score)
    
    print("Experiment run completed")
    
    # print best score and best parameters
    print("Best Score: ", best_score)
    print("Best Parameters: ", best_params)
    
    