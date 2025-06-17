import logging
import mlflow
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as f
import pandas as pd


def score_model(
    master_table: SparkDataFrame,
    end_date: str,
    model
) -> SparkDataFrame:

    logging.info(f"Iniciando scoring para o dia {end_date}")

    model_name = "xgboost_model_cartola"

    model_uri = f"models:/{model_name}/Production"
    logging.info(f"Carregando modelo de {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    master_table_pd = master_table.toPandas()  # type: ignore

    X = master_table_pd.drop(
        ['nr_cpf_cnpj', 'flag_bloqueio'], axis=1
    )
  
    y_pred_proba = model.predict(X)

    if isinstance(y_pred_proba, pd.DataFrame):
        y_pred_proba = y_pred_proba.values.flatten()

    y_pred_label = (y_pred_proba >= 0.5).astype(int)


    scored_df = master_table.select("nr_cpf_cnpj") \
        .withColumn("probabilidade_inadimplente", f.array(*[f.lit(float(p)) for p in y_pred_proba])) \
        .withColumn("flag_predito", f.array(*[f.lit(int(p)) for p in y_pred_label]))

    logging.info(f"Scoring concluído para {end_date}")

    return scored_df