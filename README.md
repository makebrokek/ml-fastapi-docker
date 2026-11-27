# ML Model Deployment with FastAPI, Docker, MLflow and Streamlit

REST API service serving an Iris classification model (`RandomForestClassifier`), with:
- a **FastAPI** backend for predictions,
- a **Streamlit** frontend for interactive use,
- **MLflow** for experiment tracking and model registry,
- everything containerized with **Docker** / **docker-compose**.

## Project Structure
- `train.py` — trains the model, logs params/metrics/artifacts to MLflow, registers the model in the MLflow Model Registry, and saves `model.joblib` for the API
- `main.py` — FastAPI application with `GET /` and `POST /predict`
- `streamlit_app.py` — Streamlit UI that calls the FastAPI `/predict` endpoint
- `model.joblib` — serialized model artifact used by the API
- `requirements.txt` — Python dependencies
- `Dockerfile` — container build instructions for the API (and the training job)
- `Dockerfile.streamlit` — container build instructions for the Streamlit frontend
- `docker-compose.yml` — orchestrates MLflow server, training job, API and frontend

## Run everything with Docker Compose (recommended)
```bash
# 1. Start the MLflow tracking server
docker compose up -d mlflow

# 2. Train the model (logs to MLflow + registers it in the Model Registry)
docker compose run --rm train

# 3. Start the API and the Streamlit frontend
docker compose up -d api streamlit
```

- MLflow UI:   http://localhost:5000
- FastAPI docs: http://localhost:8000/docs
- Streamlit app: http://localhost:8501

## Run locally without Docker
```bash
pip install -r requirements.txt

# start a local MLflow server in one terminal
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --serve-artifacts \
  --artifacts-destination ./mlruns \
  --default-artifact-root mlflow-artifacts:/

# in another terminal
python train.py                 # trains + logs to MLflow + registers the model
uvicorn main:app --reload       # starts the API on http://localhost:8000

# in another terminal
API_URL=http://localhost:8000/predict streamlit run streamlit_app.py
```

## MLflow: what gets tracked
Each run of `train.py` creates an MLflow run under the `iris-classification` experiment with:
- **Parameters:** `n_estimators`, `random_state`
- **Metrics:** `accuracy`, `f1_score`
- **Artifacts:** the serialized model (`model.joblib` and the MLflow sklearn flavor)

It also registers/updates a model named `iris-rf-classifier` in the **Model Registry**, creating
a new version on every training run. Check the "Models" tab in the MLflow UI to see versions and
move one to a stage (e.g. Staging / Production) if needed.
