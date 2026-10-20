<div align="center">
  <img src="frontend/public/logo.png" alt="ReefEye logo" width="112" />
  <h1>ReefEye</h1>
  <p><strong>A closer look beneath the surface.</strong></p>
  <p>Coral reef image analysis with deep learning, confidence-guided review, and survey management.</p>
  <p>
    <a href="#getting-started">Getting started</a> ·
    <a href="docs/MODEL_CARD.md">Model card</a> ·
    <a href="https://github.com/Maram1alzahrani/ReefEye/releases/tag/model-v1">Download weights</a>
  </p>
</div>

---

ReefEye turns reef photographs into inspectable point classifications. Organize monitoring sites and surveys, analyze image crops with a Vision Transformer, inspect uncertain predictions, and export confirmed review decisions alongside the original model output.

The application connects point-level image analysis with a human review workflow.

## Capabilities

| Area | What you can do |
| --- | --- |
| Site management | Record monitoring sites and optional coordinates; view mapped sites. |
| Survey workspace | Group photographs by site and survey, with date, depth, visibility, and notes. |
| Point analysis | Choose a 5 × 5, 7 × 7, or 10 × 10 grid, with up to 25, 49, or 100 sampled points. |
| Prediction inspection | Select image markers, inspect local crops, and filter points by class or review status. |
| Human review | Confirm or correct labels, move through flagged points, and undo the latest review. |
| Reporting | Inspect class distributions and export point-level CSV results. |
| Local persistence | Keep the workspace, photographs, predictions, and review decisions in browser IndexedDB. |

## How it works

1. **Create a site and survey.** Add survey metadata and upload reef photographs.
2. **Analyze a photograph.** The backend samples image locations, extracts 224 × 224 pixel crops, and predicts one label per crop.
3. **Inspect and review.** Predictions below the configured confidence threshold are flagged. Confirm a prediction or select a corrected label.
4. **Export the results.** CSV files retain the original prediction and confidence, the current confirmed label, and review status.

Confirmed review decisions are included in CSV exports.

## Model at a glance

| Property | Released configuration |
| --- | --- |
| Architecture | ViT-B/16 (`vit_b_16`) |
| Task | Semantic-mask-derived point crop classification |
| Dataset recorded in configuration | [josauder/coralscapesV2](https://huggingface.co/datasets/josauder/coralscapesV2) |
| Input crop | 224 × 224 RGB pixels |
| Output | Seven classification labels |
| Confidence | Temperature-scaled softmax; review threshold approximately 67.82% |
| Split unit | Dive site: 18 training, 9 validation, 8 calibration, 6 test sites |
| Inference device | CPU in the current backend |
| Weights | [Model v1](https://github.com/Maram1alzahrani/ReefEye/releases/tag/model-v1), approximately 327 MiB |

**Labels:** Acropora, Background, Galaxea, Other Coral, Pocillopora, Porites, and Stylophora. Background and Other Coral provide contextual classification categories.

See the [model card](docs/MODEL_CARD.md) for preprocessing, evaluation configuration, and confidence interpretation.

## Getting started

### Requirements

- Git.
- Python 3.11 for the backend.
- Node.js 22.12+ with npm, or another version supported by [Vite](https://vite.dev/guide/).
- A modern browser with IndexedDB support.

### 1. Clone the repository

```bash
git clone https://github.com/Maram1alzahrani/ReefEye.git
cd ReefEye
```

### 2. Download the model

From the repository root:

```bash
curl --fail --location --retry 3 \
  "https://github.com/Maram1alzahrani/ReefEye/releases/download/model-v1/reef_point_classifier.pt" \
  --output backend/model/reef_point_classifier.pt
```

Alternatively, download the checkpoint from the [release page](https://github.com/Maram1alzahrani/ReefEye/releases/tag/model-v1) and save it as `backend/model/reef_point_classifier.pt`.

<details>
<summary>Verify the checkpoint</summary>

Expected SHA-256:

```text
d865da5662ecd676904e46cdfb0f8e211c24e8bcb8384c2b31e83754a42a0319
```

On macOS:

```bash
shasum -a 256 backend/model/reef_point_classifier.pt
```

The backend checks that the checkpoint architecture and ordered class list match `backend/model/config.json`.

</details>

### 3. Start the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

The checkpoint must be present before startup. After `Application startup complete` appears, open:

- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).
- Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The health response should include `"status": "ready"`.

On Windows, use `py -3.11 -m venv .venv` to create the environment. Activate it with `.venv\Scripts\activate.bat` in Command Prompt or `.\.venv\Scripts\Activate.ps1` in PowerShell.

### 4. Start the frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite and keep both processes running. The frontend currently connects to `http://127.0.0.1:8000`.

### Build the frontend

```bash
cd frontend
npm run build
```

The production frontend is written to `frontend/dist/`. A working deployment also requires the inference backend, an appropriate API URL, and CORS configuration for the deployed frontend origin.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Report loaded model architecture and classes. |
| POST | `/predict` | Analyze an uploaded image using a sampling grid. |

`/predict` accepts multipart fields `file` and `grid_size`. The API accepts integer grid sizes from 2 to 10; the interface provides 5, 7, and 10.

Image limits: at least 224 × 224 pixels, at most 20 MiB, and at most 25 megapixels. Returned coordinates use `row = y` and `column = x` in the EXIF-corrected image.

## Results and storage

Each exported point includes site and survey metadata, image filename, pixel coordinates, original prediction, original confidence, original review flag, threshold, current label, and review status.

- **Model prediction:** an unreviewed prediction above the review threshold.
- **Needs review:** a flagged prediction awaiting confirmation.
- **Reviewed:** a label explicitly confirmed by the user.

Reviewing a point preserves the original prediction and confidence alongside the confirmed label.

The workspace is stored in IndexedDB within the current browser and origin. CSV exports provide point-level tabular results with survey metadata and confirmed review decisions.

Images are sent to the configured backend and processed in memory for inference. Mapped sites use an OpenStreetMap embed with the selected coordinates.

## Technology

| Layer | Stack |
| --- | --- |
| Interface | React, TypeScript, Vite |
| Workspace storage | IndexedDB |
| Inference API | FastAPI, Uvicorn |
| Model execution | PyTorch, torchvision |
| Image processing | Pillow, NumPy |

## Repository guide

| Path | Contents |
| --- | --- |
| `frontend/src/App.tsx` | Survey workspace, analysis, review, and export workflows |
| `frontend/src/App.css` | Application styles and responsive layouts |
| `backend/main.py` | Model startup, health check, and prediction endpoint |
| `backend/model/config.json` | Class order, preprocessing, calibration, and split metadata |
| `backend/requirements.txt` | Python dependencies |
| `docs/MODEL_CARD.md` | Model provenance, evaluation configuration, and intended use |
