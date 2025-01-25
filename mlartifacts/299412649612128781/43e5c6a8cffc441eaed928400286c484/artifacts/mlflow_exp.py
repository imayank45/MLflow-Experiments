import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# load the datasets
wine = load_wine()

# split the dataset into features (X) and target (y)
X = wine.data
y = wine.target

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# define the parameters for random forest model
max_depth = 8
n_estimators = 8

# mention experiment name
mlflow.set_experiment('MLOps Exp1')

# mlflow experiment
with mlflow.start_run():
    
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy_score = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric('accuracy_score', accuracy_score)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    
    
    
    # confusion matrix plotting
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.savefig("confusion_matrix.png")
    
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    
    print("Accuracy Score: ", accuracy_score)