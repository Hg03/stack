import mlflow
import dagshub
class Infer:
    def __init__(self):
        dagshub.init(repo_owner='Hg03', repo_name='stack', mlflow=True)
        self.model_artifact_path = 'runs:/69ecb7a277ec466cb824e64ff3d9ebf3/models/svc.joblib'
        self.loaded_model = mlflow.sklearn.load_model(model_uri = self.model_artifact_path)

    def single_inference(self, payload: dict):
        pass
    
    def batch_inference(self, payload: dict):
        pass