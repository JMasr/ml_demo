"""
Created on 11/09/2023
@author: José M. Ramírez
contact:jmramirez@gtm.uvigo.es

Main script of the ML-Demos project.
"""

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    # Load a sample dataset and split it into train and test.
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create a models with a set of parameters
    model_params = {"n_estimators": 10, "max_depth": 5, "max_features": "auto", "random_state": 42}
    rf = RandomForestRegressor(n_estimators=model_params.get("n_estimators"),
                               max_depth=model_params.get("max_depth"),
                               max_features=model_params.get("max_features"),
                               random_state=model_params.get("random_state"))

    # Train the model
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)

    # Calculate some metrics
    mae = np.mean(abs(predictions - y_test))
    print(f"Mean Absolute Error: {round(mae, 2)}")

    auc = roc_auc_score(y_test, predictions)
    print(f'AUC: {round(auc, 2)}')

    # Create a dict with the metrics
    metrics = {"mae": mae, "auc": auc}

    # MLFlow section
    port = 5000
    mlflow.set_tracking_uri(f'http://localhost:{port}')

    # Define the experiment
    experiment_name = f"diabetes_demo"
    mlflow.set_experiment(experiment_name)

    num_fold = 0
    model_name = "RandomForestRegressor"
    seed = model_params.get("random_state")

    # Log the parameters
    with mlflow.start_run(run_name=f'{model_name}_{num_fold}_seed-{seed}'):
        # Log the parameters
        mlflow.log_param('model', model_name)
        mlflow.log_param("fold", num_fold + 1)
        mlflow.log_param("seed", model_params.get("random_state"))
        mlflow.log_param("max_depth", model_params.get("max_depth"))
        mlflow.log_param("max_features", model_params.get("max_features"))
        mlflow.log_param("n_estimators", model_params.get("n_estimators"))

        # Log the metrics
        mlflow.log_metrics(metrics)
