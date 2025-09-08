import mlflow

class Infer:
    def __init__(self):
        self.model_artifact_path = 'mlflow-artifacts:/623bdbd0c8d140538ffa61707f368bbb/c74a5572f3ec4e66b835d6146aecd48a/artifacts/models/knn.joblib'
        self.loaded_model = mlflow.sklearn.load_model(model_uri = self.model_artifact_path)

    def single_inference(self, payload: dict):
        pass
    
    def batch_inference(self, payload: dict):
        pass