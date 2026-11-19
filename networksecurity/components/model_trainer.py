
import os
import sys

from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)

from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

import mlflow
import mlflow.sklearn
import dagshub


# ---------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")


# ---------------------------------------------------------
# Validate MLflow Credentials
# ---------------------------------------------------------

if not MLFLOW_TRACKING_URI:
    raise EnvironmentError("MLFLOW_TRACKING_URI is missing in .env")

if not MLFLOW_TRACKING_USERNAME:
    raise EnvironmentError("MLFLOW_TRACKING_USERNAME is missing in .env")

if not MLFLOW_TRACKING_PASSWORD:
    raise EnvironmentError("MLFLOW_TRACKING_PASSWORD is missing in .env")


# ---------------------------------------------------------
# Configure MLflow Environment
# ---------------------------------------------------------

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME

os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD


# ---------------------------------------------------------
# Initialize DagsHub
# ---------------------------------------------------------

dagshub.init(
    repo_owner="vogulam2306",
    repo_name="networksecurity",
    mlflow=True
)

logging.info("DagsHub and MLflow initialized successfully.")


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    # -----------------------------------------------------
    # MLflow Tracking
    # -----------------------------------------------------

    def track_mlflow(self, best_model, classificationmetric):

        try:

            with mlflow.start_run():

                # F1 Score
                mlflow.log_metric(
                    "f1_score",
                    float(classificationmetric.f1_score)
                )

                # Precision
                mlflow.log_metric(
                    "precision",
                    float(classificationmetric.precision_score)
                )

                # Recall
                mlflow.log_metric(
                    "recall_score",
                    float(classificationmetric.recall_score)
                )

                # Log model parameters
                for key, value in best_model.get_params().items():

                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(key, value)

                # Log trained model
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="model",
                    skops_trusted_types=[
                        "sklearn.tree._tree.Tree"
                    ]
                )

            logging.info("MLflow tracking completed successfully.")

        except Exception as e:

            logging.warning(
                f"MLflow tracking failed: {e}"
            )

    # -----------------------------------------------------
    # Model Training
    # -----------------------------------------------------

    def train_model(self, X_train, y_train, x_test, y_test):

        try:

            models = {
                "Random Forest": RandomForestClassifier(verbose=1),

                "Decision Tree": DecisionTreeClassifier(),

                "Gradient Boosting": GradientBoostingClassifier(verbose=1),

                "Logistic Regression": LogisticRegression(verbose=1),

                "AdaBoost": AdaBoostClassifier(),
            }

            params = {

                "Decision Tree": {
                    "criterion": [
                        "gini",
                        "entropy",
                        "log_loss"
                    ],
                },

                "Random Forest": {
                    "n_estimators": [
                        8,
                        16,
                        32,
                        128,
                        256
                    ]
                },

                "Gradient Boosting": {
                    "learning_rate": [
                        .1,
                        .01,
                        .05,
                        .001
                    ],

                    "subsample": [
                        0.6,
                        0.7,
                        0.75,
                        0.85,
                        0.9
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                "Logistic Regression": {},

                "AdaBoost": {
                    "learning_rate": [
                        .1,
                        .01,
                        .001
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                }

            }

            logging.info("Starting model evaluation.")

            # Model evaluation and hyperparameter tuning
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=params
            )

            if not model_report:
                raise ValueError(
                    "Model evaluation returned an empty report."
                )

            # Get best model score
            best_model_score = max(
                sorted(model_report.values())
            )

            # Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(
                f"Best model name: {best_model_name}"
            )

            logging.info(
                f"Best model score: {best_model_score}"
            )

            # -------------------------------------------------
            # Training Metrics
            # -------------------------------------------------

            y_train_pred = best_model.predict(X_train)

            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred
            )

            logging.info(
                f"Training metrics: {classification_train_metric}"
            )

            # MLflow training experiment
            self.track_mlflow(
                best_model,
                classification_train_metric
            )

            # -------------------------------------------------
            # Testing Metrics
            # -------------------------------------------------

            y_test_pred = best_model.predict(x_test)

            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred
            )

            logging.info(
                f"Testing metrics: {classification_test_metric}"
            )

            # MLflow testing experiment
            self.track_mlflow(
                best_model,
                classification_test_metric
            )

            # -------------------------------------------------
            # Load Preprocessor
            # -------------------------------------------------

            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            # -------------------------------------------------
            # Save Trained Model
            # -------------------------------------------------

            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            os.makedirs(
                model_dir_path,
                exist_ok=True
            )

            Network_Model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=Network_Model
            )

            # -------------------------------------------------
            # Model Pusher
            # -------------------------------------------------

            save_object(
                "final_model/model.pkl",
                best_model
            )

            logging.info(
                "Model and preprocessor saved successfully."
            )

            # -------------------------------------------------
            # Model Trainer Artifact
            # -------------------------------------------------

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,

                train_metric_artifact=classification_train_metric,

                test_metric_artifact=classification_test_metric
            )

            logging.info(
                f"Model trainer artifact: {model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:

            raise NetworkSecurityException(e, sys) from e

    # -----------------------------------------------------
    # Initiate Model Trainer
    # -----------------------------------------------------

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:

            logging.info(
                "Entered model trainer component."
            )

            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            # Load training and testing arrays
            train_arr = load_numpy_array_data(
                train_file_path
            )

            test_arr = load_numpy_array_data(
                test_file_path
            )

            # Split independent and dependent variables
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Train model
            model_trainer_artifact = self.train_model(
                x_train,
                y_train,
                x_test,
                y_test
            )

            logging.info(
                "Model trainer component completed successfully."
            )

            return model_trainer_artifact

        except Exception as e:

            raise NetworkSecurityException(e, sys) from e
