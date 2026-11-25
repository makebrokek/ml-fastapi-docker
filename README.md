# ML Model Deployment with FastAPI and Docker

REST API service serving an Iris classification model (`RandomForestClassifier`) built with FastAPI and containerized with Docker.

## Project Structure
- `train.py` — trains the classification model and saves `model.joblib`
- `main.py` — FastAPI application with `GET /` and `POST /predict`
- `model.joblib` — serialized model artifact
- `requirements.txt` — Python dependencies
- `Dockerfile` — container build instructions

## Run Locally
```bash
pip install -r requirements.txt
python train.py
uvicorn main:app --reload