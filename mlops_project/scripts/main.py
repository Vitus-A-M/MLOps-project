# mlops_project/scripts/main.py

import datetime
from mlops_project.data.load import pull_data, load_raw_data
from mlops_project.data.preprocess import preprocess
from mlops_project.features.build_features import build_features
from mlops_project.models.train import train_models
from mlops_project.models.evaluate import evaluate_model, compare_models


def main():
    
    pull_data()
    data = load_raw_data()
    data = preprocess(data)
    
   #Feature engineering
    data = build_features(data)
    
    #Model training
    experiment_name = datetime.datetime.now().strftime("%Y_%B_%d")
    xgb_model, lr_model, X_test, y_test = train_models(data, experiment_name)
    
    #Evaluating
    results = {
        "xgboost": evaluate_model(xgb_model.best_estimator_, X_test, y_test),
        "logistic_regression": evaluate_model(lr_model.best_estimator_, X_test, y_test)
    }
    
    # Best model
    best = compare_models(results)
    print(f"Pipeline tamamlandı! En iyi model: {best}")


if __name__ == "__main__":
    main()