import sys
import os

import certifi
import pymongo
import pandas as pd

from dotenv import load_dotenv

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

ca = certifi.where()

mongo_db_url = os.getenv("MONGODB_URL_KEY")

if not mongo_db_url:
    raise EnvironmentError("MONGODB_URL_KEY is missing")

# ==========================================
# PROJECT IMPORTS
# ==========================================

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# ==========================================
# MONGODB CONNECTION
# ==========================================

client = pymongo.MongoClient(
    mongo_db_url,
    tlsCAFile=ca,
    serverSelectionTimeoutMS=10000
)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# ==========================================
# FASTAPI APPLICATION
# ==========================================

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# ==========================================
# TRAINING ROUTE
# ==========================================

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

        return Response("Training is successful")

    except Exception as e:
        raise NetworkSecurityException(e, sys)

# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.post("/predict")
async def predict_route(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        df = pd.read_csv(file.file)

        preprocesor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        network_model = NetworkModel(
            preprocessor=preprocesor,
            model=final_model
        )

        y_pred = network_model.predict(df)

        df["predicted_column"] = y_pred

        os.makedirs("prediction_output", exist_ok=True)

        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped")

        return templates.TemplateResponse(
            "table.html",
            {
                "request": request,
                "table": table_html
            }
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)

# ==========================================
# APPLICATION STARTUP
# ==========================================

if __name__ == "__main__":
    app_run(
        app,
        host="0.0.0.0",
        port=8080
    )