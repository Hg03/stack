import polars as pl
from omegaconf import DictConfig
from stack.data_pipeline.hopsworks_implementation import init_fs


class DataPipeline:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        # self.fs = init_fs()
    def extract(self):
        print("Extracting data...")
        self.raw_data = pl.read_csv(self.config.dataset.path.raw)
        self.raw_data.columns = [col.lower() for col in self.raw_data.columns]
        self.raw_data = self.raw_data.select(pl.col(self.config.dataset.features.to_keep))

    def transform(self):
        print("Transforming data...")
        X, y = self.raw_data.select(pl.col(self.config.preprocessing.features.X)), self.raw_data.select(pl.col(self.config.preprocessing.features.y))
        
    def load(self):
        print("Loading data...")
    def run(self):
        self.extract()
        self.transform()
        self.load()
        return True