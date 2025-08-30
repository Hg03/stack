from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig
import numpy as np
from sklearn.compose import make_column_transformer, make_column_selector
import polars as pl

def merge_dfs(X_train, X_test, y_train, y_test):
    # Combine features and targets
    X_full = pl.concat([X_train, X_test])
    y_full = pl.concat([y_train, y_test])

    # Combine into single dataframe
    full_processed_data = pl.concat([X_full, y_full], how="horizontal")
    return full_processed_data

def get_imputers(config: DictConfig):
    numerical_imputer = SimpleImputer(strategy=config.preprocessing.steps.impute.numeric_strategy)
    categorical_imputer = SimpleImputer(strategy=config.preprocessing.steps.impute.categorical_strategy)
    return numerical_imputer, categorical_imputer

def get_encoder():
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    return encoder

def postprocessor(df):
    df.columns = [col[col.rfind('__')+2:] for col in df.columns]
    return df


def get_preprocessor(config: DictConfig):
    numerical_imputer, categorical_imputer = get_imputers(config)
    encoder = get_encoder()
    imputer = make_column_transformer(
        (numerical_imputer, make_column_selector(dtype_include=np.number)),
        (categorical_imputer, make_column_selector(dtype_include=object)),
        remainder='passthrough').set_output(transform="pandas")
    encoder = make_column_transformer(
        (encoder, make_column_selector(dtype_include=object)),
        remainder='passthrough').set_output(transform="pandas")
    return Pipeline(steps=[('imputer', imputer), ('encoder', encoder), ('postprocess', FunctionTransformer(postprocessor))]).set_output(transform="pandas")
    
    
    
    