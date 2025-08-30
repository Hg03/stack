import polars as pl
import joblib
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from stack.utils.hopsworks_implementation import get_fs, get_fg, add_feature_descriptions
from stack.data_pipeline.preprocess_implementation import get_preprocessor, merge_dfs


class DataPipeline:
    def __init__(self, config: DictConfig, fs: any) -> None:
        self.config = config
        self.fs = fs
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
        self.training_full_data = merge_dfs(self.X_train, self.y_train)
        self.testing_full_data = merge_dfs(self.X_test, self.y_test)
        self.training_full_data = self.training_full_data.with_row_index("id").with_columns(pl.col("id").cast(pl.Int64))
        self.testing_full_data = self.testing_full_data.with_row_index("id").with_columns(pl.col("id").cast(pl.Int64))
        self.training_full_data.write_csv(self.config.dataset.path.processed_train)
        self.testing_full_data.write_csv(self.config.dataset.path.processed_test)
        joblib.dump(preprocessor, self.config.model.path.preprocessor)

    def load(self):
        print("Loading data...")
        self.training_fg, self.testing_fg = get_fg(self.fs, self.config)
        self.training_fg.insert(self.training_full_data.to_pandas())
        self.testing_fg.insert(self.testing_full_data.to_pandas())
        add_feature_descriptions(self.training_fg, self.config)
        add_feature_descriptions(self.testing_fg, self.config)

    def run(self):
        self.extract()
        self.transform()
        if self.fs: self.load()
        
        return True