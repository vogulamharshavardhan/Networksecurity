
import sys
import os

import certifi
import pymongo
import pandas as pd

from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from uvicorn import run as app_run


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

ca = certifi.where()

mongo_db_url = os.getenv("MONGODB_URL_KEY")

if not mongo_db_url:
    raise EnvironmentError("MONGODB_URL_KEY is missing")


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

app = FastAPI(
    title="Network Security ML API",
    description="Network Security Prediction Application",
    version="1.0.0"
)

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

        return Response(
            content="Training is successful",
            media_type="text/plain"
        )

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

        # Validate uploaded file
        if not file.filename.lower().endswith(".csv"):
            return Response(
                content="Please upload a valid CSV file",
                status_code=400
            )

        # Read uploaded CSV
        df = pd.read_csv(file.file)

        if df.empty:
            return Response(
                content="Uploaded CSV file is empty",
                status_code=400
            )

        # Load preprocessing object
        preprocessor = load_object(
            "final_model/preprocessor.pkl"
        )

        # Load trained model
        final_model = load_object(
            "final_model/model.pkl"
        )

        # Initialize prediction model
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=final_model
        )

        # Generate predictions
        y_pred = network_model.predict(df)

        # Add predictions to DataFrame
        df["predicted_column"] = y_pred

        # Create output directory
        os.makedirs(
            "prediction_output",
            exist_ok=True
        )

        # Save prediction results
        df.to_csv(
            "prediction_output/output.csv",
            index=False
        )

        # Convert DataFrame into HTML
        table_html = df.to_html(
            classes="table table-striped",
            index=False,
            border=0
        )

        # Render HTML template
        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={
                "request": request,
                "table": table_html
            }
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)

    finally:
        await file.close()


# ==========================================
# APPLICATION STARTUP
# ==========================================

if __name__ == "__main__":

    app_run(
        app,
        host="0.0.0.0",
        port=8080
    )
