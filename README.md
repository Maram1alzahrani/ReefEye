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

## Model checkpoint

Trained model weights are not included in this repository.

Place the compatible checkpoint at:

```text
backend/model/reef_point_classifier.pt
```

Its architecture and classes must match `backend/model/config.json`.

## Run the backend

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The checkpoint is required for the backend to start.

## Run the frontend

In another terminal, from the repository root:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite.

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
