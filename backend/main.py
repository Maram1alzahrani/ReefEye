import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import vit_b_16

MODEL_DIR = Path(__file__).resolve().parent / "model"


@asynccontextmanager
async def lifespan(app: FastAPI):
    with (MODEL_DIR / "config.json").open() as f:
        config = json.load(f)

    checkpoint = torch.load(
        MODEL_DIR / "reef_point_classifier.pt",
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["classes"] != config["classes"]:
        raise RuntimeError("Model and config classes do not match")
    if checkpoint["architecture"] != config["architecture"]:
        raise RuntimeError("Model and config architecture do not match")
    if config["architecture"] != "vit_b_16":
        raise RuntimeError("Unsupported architecture")

    model = vit_b_16(weights=None, num_classes=len(config["classes"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    app.state.model = model
    app.state.config = config
    app.state.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])
    yield


app = FastAPI(title="ReefEye API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ready",
        "architecture": app.state.config["architecture"],
        "classes": app.state.config["classes"],
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    grid_size: int = Form(5, ge=2, le=10),
):
    data = file.file.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "Maximum file size is 20 MB")

    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 25_000_000:
                raise HTTPException(413, "Maximum image size is 25 megapixels")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Please upload a valid image")

    config = app.state.config
    size = int(config["crop_size"])
    half = size // 2
    width, height = image.size

    if width < size or height < size:
        raise HTTPException(
            400, f"Image must be at least {size} x {size} pixels"
        )

    rows = np.unique(
        np.linspace(half, height - (size - half), grid_size).astype(int)
    )
    columns = np.unique(
        np.linspace(half, width - (size - half), grid_size).astype(int)
    )
    coordinates = [(int(r), int(c)) for r in rows for c in columns]
    results = []

    with torch.inference_mode():
        for start in range(0, len(coordinates), 8):
            batch_coordinates = coordinates[start:start + 8]
            crops = []
            for row, column in batch_coordinates:
                left, top = column - half, row - half
                crop = image.crop((left, top, left + size, top + size))
                crops.append(app.state.transform(crop))

            logits = app.state.model(torch.stack(crops))
            probabilities = torch.softmax(
                logits / float(config["temperature"]), dim=1
            )
            confidence, predictions = probabilities.max(dim=1)

            for index, (row, column) in enumerate(batch_coordinates):
                score = float(confidence[index])
                needs_review = score < float(config["review_threshold"])
                results.append({
                    "point": start + index + 1,
                    "row": row,
                    "column": column,
                    "prediction": config["classes"][int(predictions[index])],
                    "confidence": score,
                    "needs_review": needs_review,
                    "status": (
                        "Needs review" if needs_review else "Model prediction"
                    ),
                })

    review_count = sum(point["needs_review"] for point in results)
    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        "classes": config["classes"],
        "points": results,
        "total_points": len(results),
        "review_count": review_count,
        "review_rate": review_count / len(results),
        "review_threshold": config["review_threshold"],
    }
