from stack.inference_pipeline.infer_implementations import EmployeeData
import pandas as pd
from typing import Optional
import mlflow
import dagshub

class Infer:
    def __init__(self):
        dagshub.init(repo_owner='Hg03', repo_name='stack', mlflow=True)
        self.model_artifact_path = 'runs:/69ecb7a277ec466cb824e64ff3d9ebf3/models/svc.joblib'
        self.loaded_model = mlflow.sklearn.load_model(model_uri = self.model_artifact_path)

    def make_inference(self, payload: Optional[EmployeeData]):
        payload_dict = {k: [v] for k, v in payload.model_dump().items()}
        payload_df = pd.DataFrame(payload_dict)
        print(payload_df)
        return self.loaded_model.predict(payload_df)
