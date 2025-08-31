from stack.configurations.config_validation import ModelInstance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import dagshub
import mlflow
from dotenv import load_dotenv
import os
from sklearn.pipeline import Pipeline

load_dotenv()

def get_model(preprocessor, model: str, hyperparams: dict[str, list[str]]):
    skmodel = ModelInstance().get_instance(model_name=model)
    estimator = Pipeline([("preprocessor", preprocessor), (model, skmodel)])
    return GridSearchCV(estimator=estimator, param_grid=dict(hyperparams))


def get_metrics(model, x_train, x_test, y_train, y_test):
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    return classification_report(y_test, test_pred, output_dict=True)

def log_models_and_metrics(config, model, x_train, x_test, y_train, y_test):
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_USER_TOKEN"), url=os.getenv("DAGSHUB_URL"))
    dagshub.init(repo_owner='Hg03', repo_name='stack', mlflow=True)
    with mlflow.start_run():
        cr = get_metrics(model, x_train, x_test, y_train, y_test)
        # Logging all metrics in classification_report
        mlflow.log_metric("accuracy", cr.pop("accuracy"))
        for class_or_avg, metrics_dict in cr.items():
            for metric, value in metrics_dict.items():
                mlflow.log_metric(class_or_avg + '_' + metric,value)
        mlflow.log_artifact(local_path=config.model.path.models)