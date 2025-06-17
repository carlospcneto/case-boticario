from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=["master_table",
                    "params:number_of_folds",
                    "params:threshold",
                    "params:end_date"
                    ],
            outputs="xgb_model",
            name="train_xgb_model",
            tags=["train"]
        )
    ])