# ReefEye

A coral reef image analysis application built with deep learning and computer vision.

ReefEye classifies sampled points in reef photographs, displays model confidence, and supports human review of uncertain predictions.

## Features

- Dashboard for monitoring sites, surveys, and review progress.
- Survey library with multiple photographs per survey.
- Site maps using recorded coordinates.
- Interactive point predictions and image crops.
- Human review with confirmed labels and undo.
- Classification distribution based on sampled points.
- CSV export with original predictions and review decisions.
- Local workspace persistence using IndexedDB.

## Technology

- Frontend: React, TypeScript, and Vite.
- Backend: FastAPI and PyTorch.
- Model architecture: Vision Transformer, ViT-B/16.

## Get the project

Install Git, Python 3.11, and a Node.js version supported by Vite, including npm.

```bash
git clone https://github.com/Maram1alzahrani/ReefEye.git
cd ReefEye
```

## Download the model

The trained checkpoint is available in [ReefEye Model v1](https://github.com/Maram1alzahrani/ReefEye/releases/tag/model-v1) as a separate download (approximately 327 MiB).

From the repository root, run:

```bash
curl --fail --location --retry 3 \
  "https://github.com/Maram1alzahrani/ReefEye/releases/download/model-v1/reef_point_classifier.pt" \
  --output backend/model/reef_point_classifier.pt
```

Alternatively, download `reef_point_classifier.pt` from the release page and place it at:

```text
backend/model/reef_point_classifier.pt
```

Keep the filename unchanged. The checkpoint architecture and classes must match `backend/model/config.json`.

SHA-256:

```text
d865da5662ecd676904e46cdfb0f8e211c24e8bcb8384c2b31e83754a42a0319
```

## Run the backend

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The checkpoint is required for the backend to start. Once startup completes, open http://127.0.0.1:8000/health; it should return `"status": "ready"`.

On Windows, activate the virtual environment with `.venv\\Scripts\\activate` instead of `source .venv/bin/activate`.

## Run the frontend

In another terminal, from the repository root:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite. Keep both terminal processes running.

Create a site and survey, upload a reef photograph, and run image analysis. Predictions require the backend on port 8000; the frontend alone cannot perform inference.

## Build the frontend

```bash
cd frontend
npm run build
```

## Interpretation and storage

- Classification percentages describe sampled points, not coral area coverage.
- Model confidence is not measured accuracy.
- Unreviewed predictions remain model-generated labels.
- Sites, surveys, photographs, and reviews are stored in the current browser.
- CSV exports exclude image files and unconfirmed selections.
- Clearing browser data can remove the local workspace.

This project supports reef image analysis and human review. Predictions require appropriate validation before use in scientific or conservation decisions.
