# Local setup and troubleshooting

[← Back to project README](../README.md)

### 1. Prerequisites

- Python 3.13 (the version used by the Dockerfile), pip and a virtual environment.
- A MongoDB Atlas or compatible TLS-enabled MongoDB connection. The API requires a MongoDB URI at startup, even when using a saved model for inference.
- AWS CLI and authorized AWS credentials or an IAM role if using S3 synchronization.
- A DagsHub repository and credentials only if remote experiment tracking is desired.

### 2. Extract the repository and install dependencies

Open a terminal in the directory containing `app.py`. If you downloaded the ZIP, extract it first.

**Windows PowerShell**

```powershell
cd networksecurity-main
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install python-multipart jinja2
```

**Linux / macOS**

```bash
cd networksecurity-main
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install python-multipart jinja2
```

The last two packages are required by CSV uploads and HTML templating but are not explicitly listed in the original `requirements.txt`. Add them there for repeatable builds. The bundled pickle files record scikit-learn version `1.9.1`; match that version if available or regenerate both artifacts with `main.py` using your installed version. Only load pickle files you trust.

### 3. Configure environment variables

Create a local `.env` file beside `app.py`. Replace the placeholders with your own credentials. Do not commit this file.

```dotenv
# The API and data ingestion component read this variable.
MONGODB_URL_KEY=mongodb+srv://<username>:<password>@<cluster>/

# The standalone CSV import script reads a different variable.
MONGO_DB_URL=mongodb+srv://<username>:<password>@<cluster>/

# Optional remote tracking (omit or leave empty for local training).
MLFLOW_TRACKING_URI=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
```

Use the same MongoDB connection string for both variables. URL-encode special characters in passwords and configure the Atlas network access list for your machine. The database and collection names are configured in `networksecurity/constant/training_pipeline/__init__.py` and `push_data.py`; update both locations consistently if you use different names. The original collection is `Networkdata`.

For remote tracking, set `MLFLOW_TRACKING_URI` to `https://dagshub.com/<owner>/<repo>.mlflow`, use your DagsHub username and an appropriate access token, and update `DAGSHUB_REPO_OWNER` and `DAGSHUB_REPO_NAME` in `networksecurity/components/model_trainer.py` to match your repository. Tracking is optional and initialized lazily.

### 4. Populate MongoDB (required for retraining)

```bash
python push_data.py
```

The script reads the labeled CSV and inserts 11,055 records into MongoDB. It uses `insert_many` without deduplication, so run it once against a fresh collection to avoid duplicates. Inference with the already bundled artifacts does not require importing the dataset.

### 5. Train locally (optional for inference)

```bash
python main.py
```

This runs all four ML stages and saves timestamped artifacts and the final model files. Inspect `artifact/` and the application logs to confirm completion. `main.py` is the local training route without S3 synchronization; the API training route uses the full orchestration pipeline, including AWS CLI sync calls.

### 6. Start the application

```bash
python app.py
```

Or use Uvicorn directly:

```bash
uvicorn app:app --host 127.0.0.1 --port 8080
```

Open [http://localhost:8080/docs](http://localhost:8080/docs) for interactive Swagger documentation. For local testing, prefer the loopback-only Uvicorn command; `python app.py` binds to all interfaces.

## API usage and troubleshooting

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Redirects to Swagger UI. |
| GET | `/docs` | Interactive API documentation. |
| GET | `/train` | Runs retraining and attempts S3 synchronization; may take considerable time. |
| POST | `/predict` | Accepts a multipart CSV upload and returns an HTML prediction table. |

### Batch prediction in Swagger UI

1. Open `/docs` and expand **POST /predict**.
2. Select **Try it out**.
3. Upload [`valid_data/test.csv`](../valid_data/test.csv) using the `file` parameter.
4. Select **Execute** and inspect the HTML response. To view the rendered page directly, use the curl example below.
5. Check `prediction_output/output.csv` for the server-side CSV result.

The input must contain all 30 feature columns in their original order, without `Result` or `predicted_column`. The supplied inference example has 12 rows. Empty files and non-CSV extensions are rejected by the endpoint; broader schema validation is not yet enforced.

### Batch prediction using curl

**Linux/macOS**

```bash
curl -f -X POST 'http://localhost:8080/predict' \
  -F 'file=@valid_data/test.csv' \
  -o prediction-results.html
```

**Windows PowerShell**

```powershell
curl.exe -f -X POST "http://localhost:8080/predict" -F "file=@valid_data/test.csv" -o prediction-results.html
```

Open the saved HTML file in your browser. The server also writes `prediction_output/output.csv`; the endpoint does not return a downloadable CSV. The output file is overwritten by subsequent requests.

### Retraining through the API

After populating MongoDB and configuring S3 access, invoke `GET /train` via Swagger or curl:

```bash
curl http://localhost:8080/train
```

The success response is `Training is successful`. S3 synchronization uses shell commands whose return codes are not checked, so independently verify uploads. Do not expose this unauthenticated retraining route to the public internet.

## Common issues

| Symptom | Check |
| --- | --- |
| Missing `MONGODB_URL_KEY` | Set the URI in `.env` at the repository root. |
| MongoDB connection fails | Check Atlas network access, credentials, and TLS configuration. |
| CSV upload or template startup errors | Install `python-multipart` and `jinja2`. |
| Incompatible pickle or estimator error | Match the original library versions or retrain both artifacts. |
| Remote metrics missing | Configure all three MLflow variables and DagsHub repository constants. |
| S3 artifacts missing | Verify AWS credentials, permissions, bucket name, and CLI return status. |
