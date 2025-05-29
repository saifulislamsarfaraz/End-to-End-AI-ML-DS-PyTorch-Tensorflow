from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.sdk.definitions.asset import Asset
import requests
from typing import List, Dict

## Define our task

def preprocess_data():
    print("Preprocessing data...")


# Define our task 2
def train_model():
    print("Training model...")

# Define our task 3
def evaluate_model():
    print("Evaluating model...")

# Define our task 4
def deploy_model():
    print("Deploying model...")

with DAG(
    'ml_pipeline',
    schedule='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example']
) as dag:
    ## Define the task
    preprocess = PythonOperator(task_id="preprocess_tast",python_callable=preprocess_data)
    train = PythonOperator(task_id="train_task",python_callable=train_model)
    evaluate = PythonOperator(task_id="evaluate_task",python_callable=evaluate_model)
    deploy = PythonOperator(task_id="deploy_task",python_callable=deploy_model)


    ## set Dependencies

    preprocess >> train >> evaluate >> deploy