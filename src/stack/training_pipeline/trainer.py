import polars as pl
from omegaconf import DictConfig

class TrainingPipeline:
    def __init__(self, config: DictConfig, fs: any):
        self.config = config
        self.fs = fs
    
    def get_training_data_online(self):
        try:
            self.training_fg = self.fs.get_feature_group(self.config.preprocessing.feature_store.train_fg, self.config.preprocessing.feature_store.version)
            self.testing_fg = self.fs.get_feature_group(self.config.preprocessing.feature_store.test_fg, self.config.preprocessing.feature_store.version)
            self.training_data = self.training_fg.read()
            self.testing_data = self.testing_fg.read()
        except:
            raise "Check at the hopsworks project, maybe feature groups are not present or could be a codebase error."
    
    def get_training_data_offline(self):
        try:
            self.training_data = pl.read_csv(self.config.dataset.path.processed_train)
            self.testing_data = pl.read_csv(self.config.dataset.path.processed_test)
        except:
            raise FileNotFoundError("Not able to found dataframe locally. probably you should run data pipeline first.")
    
    def train_model(self):
        print("Training model will come here")

    def run(self):
        if self.fs:
            self.get_training_data_online()
        else:
            self.get_training_data_offline

        self.train_model()
