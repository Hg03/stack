from omegaconf import DictConfig
import hsfs
import hopsworks
from dotenv import load_dotenv
import os
load_dotenv()

def get_fs():
    # project = hsfs.connection(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY")) # If hopsworks.login didn't work, use hsfs.
    return project.get_feature_store()

def get_fg(fs, config: DictConfig):
    return (fs.get_or_create_feature_group(
    name=config.preprocessing.feature_store.train_fg,
    version=config.preprocessing.feature_store.version,
    description=config.preprocessing.feature_store.description,
    primary_key=config.preprocessing.feature_store.primary_key,
    online_enabled=True
    ), fs.get_or_create_feature_group(
    name=config.preprocessing.feature_store.test_fg,
    version=config.preprocessing.feature_store.version,
    description=config.preprocessing.feature_store.description,
    primary_key=config.preprocessing.feature_store.primary_key,
    online_enabled=True
    ))

def add_feature_descriptions(fg, config: DictConfig):
    for desc in config.preprocessing.feature_store.feature_descriptions:
        fg.update_feature_description(desc["name"], desc["description"])