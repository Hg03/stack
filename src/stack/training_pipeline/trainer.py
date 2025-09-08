import polars as pl
from omegaconf import DictConfig
import joblib
from stack.training_pipeline.training_implementation import get_model, log_models_and_metrics
from sklearn.exceptions import DataConversionWarning
import warnings
import os
warnings.filterwarnings("ignore", category=DataConversionWarning)


class TrainingPipeline:
    def __init__(self, config: DictConfig, fs: any):
        self.config = config
        self.fs = fs
    
    def get_training_data_online(self):
        print("Fetching Training data from Feature Store")
        try:
            self.training_fg = self.fs.get_feature_group(self.config.preprocessing.feature_store.train_fg, self.config.preprocessing.feature_store.version)
            self.testing_fg = self.fs.get_feature_group(self.config.preprocessing.feature_store.test_fg, self.config.preprocessing.feature_store.version)
            self.training_data = pl.DataFrame(self.training_fg.read())
            self.testing_data = pl.DataFrame(self.testing_fg.read())
        except:
            raise "Check at the hopsworks project, maybe feature groups are not present or could be a codebase error."
    
    def get_training_data_offline(self):
        print("Fetching Training data from local")
        try:
            self.training_data = pl.read_csv(self.config.dataset.path.processed_train)
            self.testing_data = pl.read_csv(self.config.dataset.path.processed_test)
        except:
            raise FileNotFoundError("Not able to found dataframe locally. probably you should run data pipeline first.")
    
    def train_model(self):
        print("Training Model")
        print(f"Selected model to train: {self.config.model.passed_model}")
        if self.config.model.passed_model in self.config.model.list_of_models:
            self.X_train, self.y_train = self.training_data.select(pl.col(self.config.preprocessing.features.X)), self.training_data.select(self.config.preprocessing.features.y)
            self.X_test, self.y_test = self.testing_data.select(pl.col(self.config.preprocessing.features.X)), self.testing_data.select(self.config.preprocessing.features.y)
            preprocessor = joblib.load(self.config.model.path.preprocessor)
            model = self.config.model.passed_model
            hyperparams = self.config.model.models.get(self.config.model.passed_model)
            self.estimator = get_model(preprocessor, model, hyperparams)
            self.estimator.fit(self.X_train.to_pandas(), self.y_train.to_pandas())
            self.best_estimator = self.estimator.best_estimator_
            # joblib.dump(self.best_estimator, os.path.join(self.config.model.path.models, f"{model}.joblib"))
        else:
            raise KeyError("Please check model config and passed appropriate model name")
    
    def log_to_mlflow(self):
        print("Logging Model and Metrics")
        log_models_and_metrics(config=self.config, model=self.best_estimator, x_train=self.X_train, x_test=self.X_test, y_train=self.y_train, y_test=self.y_test)

    def run(self):
        if self.fs:
            self.get_training_data_online()
        else:
            self.get_training_data_offline()
        self.train_model()
        self.log_to_mlflow()
        return True
