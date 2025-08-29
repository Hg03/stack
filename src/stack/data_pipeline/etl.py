import duckdb as duck
from omegaconf import DictConfig

class DataPipeline:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
    def extract(self):
        print("Extracting data...")
    def transform(self):
        print("Transforming data...")
    def load(self):
        print("Loading data...")
    def run(self):
        self.extract()
        self.transform()
        self.load()
        return True