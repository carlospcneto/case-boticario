from kedro.pipeline import Pipeline, node, pipeline

from .nodes import(
    gen_financial_features,
    gen_behavioral_features,
    agg_financial_features,
    master_table
)

def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                func=gen_financial_features,
                inputs=[
                        "params:prod",
                        "params:end_date",
                        "params:days_to_evaluate",
                        "raw_pix",
                        "raw_ted",
                        "raw_doc"
                ],
                outputs="financial_features",
                name="gen_financial_features_node",
                tags=["preprocessing"]
            ),
            node(
                func=gen_behavioral_features,
                inputs=[
                        "params:prod",
                        "params:end_date",
                        "params:days_to_evaluate",
                        "raw_campanhas",
                        "raw_riscos"
                ],
                outputs="behavioral_features",
                name="gen_behavioral_features",
                tags=["preprocessing"]                    
            ),
            node(
                func=agg_financial_features,
                inputs=[
                        "params:prod",
                        "params:end_date",
                        "params:days_to_evaluate",
                        "financial_features",
                        "cartola_table"
                ],
                outputs="agg_financial_features",
                name="agg_financial_features_node",
                tags=["preprocessing"] 
            ),
            node(
                func=master_table,
                inputs=[
                    "behavioral_features",
                    "agg_financial_features"
                ],
                outputs="master_table",
                name="master_table_node",
                tags=["master"]
            )
        ]
    )

