# FastAPI DDI Inference API

This project exposes the DDI prediction model as a FastAPI service with load balancing support. It enables real-time drug-drug interaction inference using a GNN model.

## Features
- REST endpoints for DDI prediction, drug listing, and drug addition
- Load balancing via Uvicorn workers
- Model and data files must be copied from the original model directory

## Setup
1. Copy the following files from your model directory:
   - inference_api.py
   - model_loader.py
   - models.py
   - requirements.txt
   - artifacts_latest/ (directory)
   - data/ (directory)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API with load balancing:
   ```bash
   uvicorn inference_api:app --host 0.0.0.0 --port 8000 --workers 4
   ```

## Endpoints
- `/predict` (POST): Run DDI inference
- `/drugs` (GET): List available drugs
- `/add-drug` (POST): Add a new drug

## Notes
- Ensure all required model artifacts and data files are present.
- Adjust worker count as needed for your hardware.