import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/saifulislamsarfaraz/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "saifulislamsarfaraz"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "ec2739bb70ae0efb5ac3f7d2a2f0f05a3ebafb3b"

def hyperparameter_tuing(X_train,y_train,param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search

 ## Load the parameters from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    
    mlflow.set_tracking_uri("https://dagshub.com/saifulislamsarfaraz/machinelearningpipeline.mlflow")

    ##  start the MLFLOW run 
    with mlflow.start_run():
        ## spit the dataset into training and test sets
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20,random_state=42 )
        signature = infer_signature(X_train,y_train)

        params_grid = {
            'n_estimators':[100,200],
            'max_depth': [5,10,None],
            'min_samples_split':[2,5],
            'min_samples_leaf':[1,2]
        }

        grid_search=hyperparameter_tuing(X_train,y_train,params_grid)
        best_model=grid_search.best_estimator_

        ## predict and evalute the model
        y_pred = best_model.predict(X_test)
        accuracy_score=accuracy_score(y_test,y_pred)
        print(f"Accuracy: {accuracy_score}")

        ## Log Additional Metrics
        mlflow.log_metric("accuracy",accuracy_score)
        mlflow.log_param("best_n_estimators",grid_search.best_params_['n_estimators '])
        mlflow.log_param("best_max_depth",grid_search.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_split",grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf",grid_search.best_params_['min_samples_leaf'])

        ## log the confusion matrix and classification report
        conf_matrix = confusion_matrix(y_test,y_pred)
        class_report = classification_report(y_test,y_pred)

        mlflow.log_text(str(conf_matrix),"confussion matrix.txt")
        mlflow.log_text(class_report,"classification_report.txt")

        tracking_url_type_store=urlparse(mlflow.get_tracking_url()).scheme

        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model,"model",signature=signature)

        ## Create the directory to save the model
        os.mkdirs(os.path.dirname(model_path),exist_ok=True)
        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))

        print(f"Model saved to {model_path}")

if __name__=="__main__":
    train(params["data"],params["model"],params["random_state"],params["n_estimators"],params["max_depth"])
