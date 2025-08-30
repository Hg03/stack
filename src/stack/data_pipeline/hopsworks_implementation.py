import hopsworks
from dotenv import load_dotenv
import os
load_dotenv()

def init_fs():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    return project.get_feature_store()