from stack.configurations.config_validation import MainConfig
from stack.utils.hopsworks_implementation import get_fs
from stack.data_pipeline.etl import DataPipeline
from stack.training_pipeline.trainer import TrainingPipeline
from omegaconf import OmegaConf, DictConfig
import hydra

class StackPipeline:
    def __init__(self, config: DictConfig) -> None:
        self.config = OmegaConf.structured(MainConfig(**config))
        self.fs = None if self.config.pipeline.local else get_fs()
        self.data_pipeline = DataPipeline(self.config, self.fs)
        self.training_pipeline = TrainingPipeline(self.config, self.fs)
    def execute(self):
        if self.config.pipeline.type == "data":
            print("Starting the data pipeline...")
            data_pipeline_status = self.data_pipeline.run()
            return True if data_pipeline_status else False
        elif self.config.pipeline.type == "train":
            print("Starting the training pipeline...")
            training_pipeline_status = self.training_pipeline.run()
            return True if training_pipeline_status else False
        else:
            print("Starting the data pipeline...")
            data_pipeline_status = self.data_pipeline.run()
            print("Starting the training pipeline...")
            training_pipeline_status = self.training_pipeline.run()
            return True if data_pipeline_status and training_pipeline_status else False

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.pipeline.type not in ('data', 'train', 'full'):
        raise TypeError("Invalid Pipeline Type")
    pipeline = StackPipeline(cfg)
    success = pipeline.execute()
    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline execution failed!")

if __name__ == "__main__":
    main()