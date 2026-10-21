import base64
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import vit_b_16


MODEL_DIR = Path(__file__).resolve().parent / "model"


@asynccontextmanager
async def lifespan(app: FastAPI):
    with (MODEL_DIR / "config.json").open() as file:
        config = json.load(file)

    checkpoint = torch.load(
        MODEL_DIR / "reef_point_classifier.pt",
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["classes"] != config["classes"]:
        raise RuntimeError(
            "Model and config classes do not match"
        )

    if checkpoint["architecture"] != config["architecture"]:
        raise RuntimeError(
            "Model and config architecture do not match"
        )

    if config["architecture"] != "vit_b_16":
        raise RuntimeError("Unsupported architecture")

    model = vit_b_16(
        weights=None,
        num_classes=len(config["classes"]),
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.eval()

    app.state.model = model
    app.state.config = config
    app.state.model_lock = Lock()

    app.state.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            config["mean"],
            config["std"],
        ),
    ])

    yield


app = FastAPI(
    title="ReefEye API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def read_image(file: UploadFile):
    maximum_bytes = 20 * 1024 * 1024
    data = file.file.read(maximum_bytes + 1)

    if len(data) > maximum_bytes:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 20 MB",
        )

    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 25_000_000:
                raise HTTPException(
                    status_code=413,
                    detail="Maximum image size is 25 megapixels",
                )

            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image",
        )

    size = int(app.state.config["crop_size"])

    if image.width < size or image.height < size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image must be at least "
                f"{size} x {size} pixels"
            ),
        )

    return image


def png_url(image: Image.Image):
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def gradcam(model, tensor, target_index):
    """
    Compute class-specific Grad-CAM for torchvision ViT.

    Returns:
        Detached logits and a normalized heatmap.
        The heatmap is None when no positive signal exists.
    """
    captured = {}

    def capture(_module, _inputs, output):
        captured["features"] = output

    # Capture patch features before the final attention block.
    target_layer = model.encoder.layers[-1].ln_1

    handle = target_layer.register_forward_hook(
        capture
    )

    try:
        with torch.enable_grad():
            inputs = tensor.detach().requires_grad_(True)

            logits = model(inputs)
            features = captured["features"]

            gradients = torch.autograd.grad(
                outputs=logits[0, target_index],
                inputs=features,
                retain_graph=False,
                create_graph=False,
            )[0]

    finally:
        handle.remove()

    # Exclude the class token from the spatial heatmap.
    patches = features.detach()[:, 1:, :]
    patch_gradients = gradients.detach()[:, 1:, :]

    weights = patch_gradients.mean(
        dim=1,
        keepdim=True,
    )

    grid = tensor.shape[-1] // model.patch_size

    cam = (
        patches * weights
    ).sum(dim=-1).reshape(
        1,
        1,
        grid,
        grid,
    )

    cam = F.interpolate(
        cam.relu(),
        size=tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    cam = cam - cam.min()
    peak = float(cam.max())

    if not torch.isfinite(cam).all() or peak <= 1e-12:
        return logits.detach(), None

    normalized_cam = (
        cam / peak
    ).clamp(
        0,
        1,
    ).cpu().numpy()

    return logits.detach(), normalized_cam


def heatmap_image(cam):
    """
    Convert a normalized heatmap into a transparent RGBA image.
    """
    stops = np.array([
        0.0,
        0.35,
        0.65,
        1.0,
    ])

    colors = np.array([
        [25, 70, 180],
        [20, 200, 210],
        [255, 220, 70],
        [240, 65, 40],
    ])

    rgb = np.stack([
        np.interp(
            cam,
            stops,
            colors[:, channel],
        )
        for channel in range(3)
    ], axis=-1)

    alpha = (cam * 255)[..., None]

    rgba = np.concatenate([
        rgb,
        alpha,
    ], axis=-1)

    return Image.fromarray(
        rgba.astype(np.uint8)
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
    image = read_image(file)
    config = app.state.config

    size = int(config["crop_size"])
    half = size // 2
    width, height = image.size

    rows = np.unique(
        np.linspace(
            half,
            height - (size - half),
            grid_size,
        ).astype(int)
    )

    columns = np.unique(
        np.linspace(
            half,
            width - (size - half),
            grid_size,
        ).astype(int)
    )

    coordinates = [
        (int(row), int(column))
        for row in rows
        for column in columns
    ]

    results = []

    # Prevent concurrent predictions from entering explanation hooks.
    with app.state.model_lock, torch.inference_mode():
        for start in range(0, len(coordinates), 8):
            batch_coordinates = coordinates[
                start:start + 8
            ]

            crops = []

            for row, column in batch_coordinates:
                left = column - half
                top = row - half

                crop = image.crop((
                    left,
                    top,
                    left + size,
                    top + size,
                ))

                crops.append(
                    app.state.transform(crop)
                )

            batch = torch.stack(crops)
            logits = app.state.model(batch)

            probabilities = torch.softmax(
                logits / float(config["temperature"]),
                dim=1,
            )

            confidence, predictions = probabilities.max(
                dim=1
            )

            for index, (row, column) in enumerate(
                batch_coordinates
            ):
                score = float(confidence[index])

                needs_review = (
                    score
                    < float(config["review_threshold"])
                )

                prediction = config["classes"][
                    int(predictions[index])
                ]

                results.append({
                    "point": start + index + 1,
                    "row": row,
                    "column": column,
                    "prediction": prediction,
                    "confidence": score,
                    "needs_review": needs_review,
                    "status": (
                        "Needs review"
                        if needs_review
                        else "Model prediction"
                    ),
                })

    review_count = sum(
        point["needs_review"]
        for point in results
    )

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


@app.post("/explain")
def explain(
    file: UploadFile = File(...),
    row: int = Form(..., ge=0),
    column: int = Form(..., ge=0),
    target_label: str = Form(...),
):
    image = read_image(file)
    config = app.state.config

    if target_label not in config["classes"]:
        raise HTTPException(
            status_code=400,
            detail="Choose a supported model label",
        )

    size = int(config["crop_size"])
    half = size // 2

    left = column - half
    top = row - half

    if (
        left < 0
        or top < 0
        or left + size > image.width
        or top + size > image.height
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected point must contain "
                "a complete model crop"
            ),
        )

    crop = image.crop((
        left,
        top,
        left + size,
        top + size,
    ))

    tensor = app.state.transform(
        crop
    ).unsqueeze(0)

    target_index = config["classes"].index(
        target_label
    )

    with app.state.model_lock:
        logits, cam = gradcam(
            app.state.model,
            tensor,
            target_index,
        )

    probabilities = torch.softmax(
        logits / float(config["temperature"]),
        dim=1,
    )[0]

    predicted_index = int(
        probabilities.argmax()
    )

    return {
        "method": "Grad-CAM",
        "target_layer": "encoder.layers[-1].ln_1",
        "row": row,
        "column": column,
        "target_label": target_label,
        "target_confidence": float(
            probabilities[target_index]
        ),
        "prediction": config["classes"][
            predicted_index
        ],
        "has_signal": cam is not None,
        "crop_image": png_url(crop),
        "heatmap_image": (
            png_url(heatmap_image(cam))
            if cam is not None
            else None
        ),
    }