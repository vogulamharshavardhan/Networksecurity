import os
import sys
from threading import Lock
from typing import Any

from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow
import mlflow.sklearn
import dagshub


# Load local .env when present. In Docker/EC2, variables can also be injected
# directly through the container environment.
load_dotenv()

DAGSHUB_REPO_OWNER = "vogulam2306"
DAGSHUB_REPO_NAME = "networksecurity"

_mlflow_initialized = False
_mlflow_init_lock = Lock()


def configure_mlflow() -> bool:
    """
    Configure MLflow only when tracking is actually needed.

    This function is deliberately not called during module import. Therefore,
    missing MLflow credentials cannot prevent FastAPI/Uvicorn from starting.
    Returns False when MLflow credentials are not configured.
    """
    global _mlflow_initialized

    with _mlflow_init_lock:
        if _mlflow_initialized:
            return True

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not tracking_uri:
            logging.warning(
                "MLflow tracking is disabled: MLFLOW_TRACKING_URI is not set."
            )
            return False

        if not username or not password:
            logging.warning(
                "MLflow tracking is disabled: MLflow username/password "
                "environment variables are not configured."
            )
            return False

        # DagsHub uses these environment variables for authentication.
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        # Initialize DagsHub lazily, at the point tracking is first required.
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )

        # Respect the tracking URI supplied through the environment.
        mlflow.set_tracking_uri(tracking_uri)

        _mlflow_initialized = True
        logging.info("DagsHub and MLflow initialized successfully.")
        return True


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def track_mlflow(
        self,
        best_model: Any,
        train_metric: Any = None,
        test_metric: Any = None,
    ) -> None:
        """
        Log one training run with both training and testing metrics.

        MLflow is optional for application startup and model training. If its
        credentials are absent or the tracking service is unavailable, model
        training and local artifact saving can still complete.
        """
        try:
            if not configure_mlflow():
                return

            with mlflow.start_run():
                if train_metric is not None:
                    mlflow.log_metric(
                        "train_f1_score", float(train_metric.f1_score)
                    )
                    mlflow.log_metric(
                        "train_precision",
                        float(train_metric.precision_score),
                    )
                    mlflow.log_metric(
                        "train_recall", float(train_metric.recall_score)
                    )

                if test_metric is not None:
                    mlflow.log_metric(
                        "test_f1_score", float(test_metric.f1_score)
                    )
                    mlflow.log_metric(
                        "test_precision",
                        float(test_metric.precision_score),
                    )
                    mlflow.log_metric(
                        "test_recall", float(test_metric.recall_score)
                    )

                # Log scalar/string model parameters only.
                for key, value in best_model.get_params().items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(key, value)

                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )

            logging.info("MLflow tracking completed successfully.")

        except Exception:
            # Do not hide the traceback. A tracking outage should not discard
            # the locally trained model or stop artifact creation.
            logging.exception("MLflow tracking failed.")

    def train_model(self, X_train, y_train, x_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(
                    verbose=1, max_iter=1000
                ),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            logging.info("Starting model evaluation.")

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            if not model_report:
                raise ValueError(
                    "Model evaluation returned an empty report."
                )

            # Select the model with the highest evaluation score.
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info("Best model name: %s", best_model_name)
            logging.info("Best model score: %s", best_model_score)

            # Training metrics.
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )
            logging.info(
                "Training metrics: %s", classification_train_metric
            )

            # Testing metrics.
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )
            logging.info(
                "Testing metrics: %s", classification_test_metric
            )

            # Log both metric sets in one MLflow run.
            self.track_mlflow(
                best_model=best_model,
                train_metric=classification_train_metric,
                test_metric=classification_test_metric,
            )

            # Load the fitted preprocessing object.
            preprocessor = load_object(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_object_file_path
                )
            )

            # Build the combined preprocessing + model object.
            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            if model_dir_path:
                os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model,
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=network_model,
            )

            # Preserve the additional final_model artifact, but make its path
            # configurable instead of hardcoding it in the source.
            final_model_path = os.getenv(
                "FINAL_MODEL_PATH",
                os.path.join("final_model", "model.pkl"),
            )
            final_model_dir = os.path.dirname(final_model_path)
            if final_model_dir:
                os.makedirs(final_model_dir, exist_ok=True)

            save_object(final_model_path, obj=best_model)

            logging.info(
                "Model and preprocessor saved successfully."
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )

            logging.info(
                "Model trainer artifact: %s", model_trainer_artifact
            )
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Entered model trainer component.")

            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(
                x_train,
                y_train,
                x_test,
                y_test,
            )

            logging.info(
                "Model trainer component completed successfully."
            )
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
