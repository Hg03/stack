# config_loader.py
from stack.configurations.config_validation import MainConfig
from omegaconf import OmegaConf, DictConfig
import hydra

@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def sample_loader(cfg: DictConfig) -> DictConfig:
    print(OmegaConf.structured(MainConfig(**cfg)))

if __name__ == "__main__":
    sample_loader()