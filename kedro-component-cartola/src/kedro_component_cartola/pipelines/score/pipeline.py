from kedro.pipeline import Pipeline, node, pipeline
from .nodes import score_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=score_model,
            inputs=["master_table",
                    "params:end_date",
                    "xgb_model"
                    ],
            outputs="score_table",
            name="score_xgb_model",
            tags=["score"]
        )
    ])