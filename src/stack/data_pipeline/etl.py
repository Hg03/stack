import polars as pl
import joblib
import os
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from stack.data_pipeline.hopsworks_implementation import init_fs, get_feature_group, add_feature_descriptions
from stack.data_pipeline.preprocess_implementation import get_preprocessor, merge_dfs


class DataPipeline:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.fs = init_fs()
    def extract(self):
        print("Extracting data...")
        self.raw_data = pl.read_csv(self.config.dataset.path.raw)
        self.raw_data.columns = [col.lower() for col in self.raw_data.columns]
        self.raw_data = self.raw_data.select(pl.col(self.config.dataset.features.to_keep))

    def transform(self):
        print("Transforming data...")
        X, y = self.raw_data.select(pl.col(self.config.preprocessing.features.X)), self.raw_data.select(pl.col(self.config.preprocessing.features.y))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.config.preprocessing.test_size, random_state=self.config.preprocessing.random_state, stratify=y)
        preprocessor = get_preprocessor(self.config)
        self.X_train = pl.DataFrame(preprocessor.fit_transform(self.X_train.to_pandas()))
        self.X_test = pl.DataFrame(preprocessor.transform(self.X_test.to_pandas()))
        self.full_data = merge_dfs(self.X_train, self.X_test, self.y_train, self.y_test)
        self.full_data = self.full_data.with_row_index("id").with_columns(pl.col("id").cast(pl.Int64))
        self.full_data.write_csv(self.config.dataset.path.processed)
        joblib.dump(preprocessor, self.config.model.path.preprocessor)

    def load(self):
        print("Loading data...")
        self.fg = get_feature_group(self.fs, self.config)
        self.fg.insert(self.full_data.to_pandas())
        add_feature_descriptions(self.fg, self.config)

    def run(self):
        self.extract()
        self.transform()
        self.load()
        return True