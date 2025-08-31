from stack.configurations.config_validation import ModelInstance
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def get_model(preprocessor, model: str, hyperparams: dict[str, list[str]]):
    skmodel = ModelInstance().get_instance(model_name=model)
    estimator = Pipeline([("preprocessor", preprocessor), (model, skmodel)])
    return GridSearchCV(estimator=estimator, param_grid=dict(hyperparams))