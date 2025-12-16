# mlops_project/scripts/main.py

import datetime
import os
import pickle
import json
    
from mlops_project.data.load import pull_data, load_raw_data
from mlops_project.data.preprocess import preprocess
from mlops_project.features.build_features import build_features
from mlops_project.models.train import train_models
from mlops_project.models.evaluate import evaluate_model, compare_models

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    pull_data()
    data = load_raw_data()
    data = preprocess(data)
    
   #Feature engineering
    data = build_features(data)
    
    #Model training
    experiment_name = datetime.datetime.now().strftime("%Y_%B_%d")
    xgb_model, lr_model, X_test, y_test = train_models(data, experiment_name)
    
    best = compare_models(results)
    print(f"Pipeline completed! Best model: {best}")
     
    #Evaluating
    results = {
        "xgboost": evaluate_model(xgb_model.best_estimator_, X_test, y_test),
        "logistic_regression": evaluate_model(lr_model.best_estimator_, X_test, y_test)
    }
    
    # Best model
    best_model = lr_model.best_estimator_ if best == "logistic_regression" else xgb_model.best_estimator_
    with open(f"{output_dir}/model", "wb") as f:
        pickle.dump(best_model, f)
    
    # Save both models
    with open(f"{output_dir}/xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model.best_estimator_, f)
    
    with open(f"{output_dir}/logistic_regression_model.pkl", "wb") as f:
        pickle.dump(lr_model.best_estimator_, f)
    
    # Save results (convert NumPy types to JSON-serializable)
    results_serializable = convert_to_serializable(results)
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    # Save best model info
    with open(f"{output_dir}/best_model.txt", "w") as f:
        f.write(f"Best model: {best}\n")
        f.write(f"Experiment: {experiment_name}\n")
    
    print(f"Artifacts saved to {output_dir}/")

if __name__ == "__main__":
    main()