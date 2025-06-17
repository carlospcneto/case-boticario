import logging, mlflow, shap
import mlflow.xgboost
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame as SparkDataFrame
from functools import partial
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from kedro_component_cartola.utils.auxiliar_functions import back_in_time


def train_model(master_table: SparkDataFrame,
                number_of_folds: int,
                threshold: float,
                end_date: str):

    roc_auc_last_run = 0.0

    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name("kedro_component_cartola")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],  # type: ignore
            filter_string=f"attributes.status = 'FINISHED' and tags.date = '{back_in_time(end_date, days=1)}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        last_run = runs[0]
        roc_auc_last_run = last_run.data.metrics.get("roc_auc")
        run_id_last = last_run.info.run_id
        logging.info(f"✅ Última run ID: {run_id_last}, ROC-AUC: {roc_auc_last_run}")
    except Exception:
        roc_auc_last_run = 0.0
        run_id_last = None
    

    master_table_pd = master_table.toPandas() # type: ignore

    X = master_table_pd.drop(['nr_cpf_cnpj', 'flag_bloqueio'], axis=1) 
    y = master_table_pd['flag_bloqueio']

    if roc_auc_last_run is None or roc_auc_last_run < threshold:

        logging.info("Definido espaço de busca:")
        space_xgb = {
            'objective': 'binary:logistic',
            'random_state': 42,
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),  
            'max_depth': hp.quniform('max_depth', 3, 10, 1),            
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'gamma': hp.uniform('gamma', 0.5, 5),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0), 
            'reg_alpha': hp.uniform('reg_alpha', 0, 5),                   
            'reg_lambda': hp.uniform('reg_lambda', 0, 5),                 
            'eval_metric': 'auc'
        }
        logging.info(space_xgb)

        def objective(params, X, y, number_of_folds=3):

            params['n_estimators'] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])

            clf = XGBClassifier(**params)

            kfold = KFold(n_splits=number_of_folds, shuffle=True, random_state=42)

            score = cross_val_score(
                clf, X, y, cv=kfold, scoring='roc_auc', n_jobs=-1
            ).mean()

            logging.info("\nParâmetros utilizados:", params)
            logging.info(f"Mean ROC-AUC ({number_of_folds}-fold CV): {score}\n")

            return {
                'loss': 1 - score,
                'status': STATUS_OK,
                'params': params
            }

        trials = Trials()

        objective_xgb = partial(
            objective,
            X=X,
            y=y,
            number_of_folds=number_of_folds
        )

        best_params = fmin(
            fn=objective_xgb,
            space=space_xgb,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials,
            rstate=np.random.default_rng(42)
        )

        best_params['n_estimators'] = int(best_params['n_estimators']) # type: ignore
        best_params['max_depth'] = int(best_params['max_depth']) # type: ignore
        best_params['objective'] = 'binary:logistic' # type: ignore
        best_params['eval_metric'] = 'auc' # type: ignore
        best_params['random_state'] = 42 # type: ignore

        final_model = XGBClassifier(**best_params) # type: ignore
        final_model.fit(X, y)

        y_pred_proba = final_model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)

        
        mlflow.log_params(best_params)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.set_tag("date", end_date)

        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="xgboost_model_cartola"
        )

        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="violin", show=False)
        shap_plot_path = "shap_summary_plot.png"
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()

        mlflow.log_artifact(shap_plot_path)
    else:
        final_model = mlflow.pyfunc.load_model(f"runs:/{run_id_last}/model")

        y_pred_proba = final_model.predict(X)[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)

        logging.info(f"ROC-AUC do dia: {roc_auc}")

        
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.set_tag("date", end_date)

        # Logar novamente o modelo só como referência (opcional)
        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="xgboost_model_cartola",

        )

    client = MlflowClient()
    latest_version = client.get_latest_versions("xgboost_model_cartola", stages=["None"])[0].version

    # Promover diretamente para Production
    client.transition_model_version_stage(
        name="xgboost_model_cartola",
        version=latest_version,
        stage="Production"
)

    return final_model