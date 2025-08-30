from stack.configurations.config_validation import MainConfig
from stack.utils.hopsworks_implementation import get_fs
from stack.data_pipeline.etl import DataPipeline
from omegaconf import OmegaConf, DictConfig
import hydra

class StackPipeline:
    def __init__(self, config: DictConfig) -> None:
        self.config = OmegaConf.structured(MainConfig(**config))
        self.fs = None #get_fs()
        self.data_pipeline = DataPipeline(self.config, self.fs)
    def execute(self):
        if self.config.pipeline.type == "data":
            print("Starting the data pipeline...")
            data_pipeline_status = self.data_pipeline.run()
            return True
        elif self.config.pipeline.type == "train":
            pass
        else:
            print("Starting the data pipeline...")
            data_pipeline_status = self.data_pipeline.run()
        

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