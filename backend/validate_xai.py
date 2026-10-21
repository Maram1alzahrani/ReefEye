import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.models import vit_b_16

from main import gradcam


FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.5)


def patch_scores(cam, grid):
    tensor = torch.tensor(
        cam,
        dtype=torch.float32,
    )[None, None]

    return F.adaptive_avg_pool2d(
        tensor,
        (grid, grid),
    ).numpy().ravel()


def compare_maps(first, second, grid):
    if first is None or second is None:
        return {
            "status": "inconclusive_no_signal",
        }

    a = patch_scores(first, grid)
    b = patch_scores(second, grid)

    if a.std() < 1e-8 or b.std() < 1e-8:
        return {
            "status": "inconclusive_constant_map",
        }

    return {
        "status": "compared",
        "pearson": float(
            np.corrcoef(a, b)[0, 1]
        ),
        "mean_absolute_difference": float(
            np.abs(a - b).mean()
        ),
        "nearly_identical": bool(
            np.allclose(
                a,
                b,
                rtol=0,
                atol=1e-5,
            )
        ),
    }


def ranked_order(scores, rng, hottest):
    # Random tie-breaking avoids a top-left selection bias.
    shuffled = rng.permutation(len(scores))

    values = (
        -scores[shuffled]
        if hottest
        else scores[shuffled]
    )

    return shuffled[
        np.argsort(values, kind="stable")
    ]


def deletion_curve(
    model,
    tensor,
    baseline,
    target,
    temperature,
    order,
    patch,
):
    grid = tensor.shape[-1] // patch
    total = grid * grid

    counts = [
        round(fraction * total)
        for fraction in FRACTIONS
    ]

    masks = torch.zeros(
        len(counts),
        total,
        dtype=tensor.dtype,
    )

    for index, count in enumerate(counts):
        masks[index, order[:count].tolist()] = 1

    masks = masks.reshape(
        -1,
        1,
        grid,
        grid,
    )

    masks = masks.repeat_interleave(
        patch,
        dim=2,
    ).repeat_interleave(
        patch,
        dim=3,
    )

    inputs = (
        tensor * (1 - masks)
        + baseline * masks
    )

    probabilities = []

    with torch.inference_mode():
        for batch in inputs.split(4):
            logits = model(batch)

            scores = torch.softmax(
                logits / temperature,
                dim=1,
            )[:, target]

            probabilities.extend(
                scores.tolist()
            )

    actual_fractions = np.array(counts) / total

    return (
        actual_fractions,
        np.array(probabilities),
    )


def partial_auc(fractions, scores):
    # Average confidence over the tested deletion range.
    widths = np.diff(fractions)

    area = np.sum(
        widths * (scores[:-1] + scores[1:]) / 2
    )

    return float(
        area / (fractions[-1] - fractions[0])
    )


def deletion_test(
    model,
    tensor,
    baseline,
    cam,
    target,
    temperature,
    trials,
    seed,
):
    patch = model.patch_size
    grid = tensor.shape[-1] // patch

    scores = patch_scores(cam, grid)
    rng = np.random.default_rng(seed)

    fractions, hot = deletion_curve(
        model,
        tensor,
        baseline,
        target,
        temperature,
        ranked_order(scores, rng, hottest=True),
        patch,
    )

    _, cold = deletion_curve(
        model,
        tensor,
        baseline,
        target,
        temperature,
        ranked_order(scores, rng, hottest=False),
        patch,
    )

    random_curves = []

    for _ in range(trials):
        _, curve = deletion_curve(
            model,
            tensor,
            baseline,
            target,
            temperature,
            rng.permutation(grid * grid),
            patch,
        )

        random_curves.append(curve)

    random_curves = np.stack(random_curves)

    random_mean = random_curves.mean(axis=0)
    hot_auc = partial_auc(fractions, hot)

    random_aucs = [
        partial_auc(fractions, curve)
        for curve in random_curves
    ]

    return {
        "fractions": fractions.tolist(),
        "original_confidence": float(hot[0]),
        "hot_confidence": hot.tolist(),
        "cold_confidence": cold.tolist(),
        "random_mean_confidence": random_mean.tolist(),
        "random_std_confidence": (
            random_curves.std(axis=0).tolist()
        ),
        "random_trial_confidence": random_curves.tolist(),
        "hot_drop_percentage_points": (
            (hot[0] - hot) * 100
        ).tolist(),
        "random_drop_percentage_points": (
            (hot[0] - random_mean) * 100
        ).tolist(),
        "hot_partial_auc": hot_auc,
        "cold_partial_auc": partial_auc(
            fractions,
            cold,
        ),
        "random_mean_partial_auc": float(
            np.mean(random_aucs)
        ),
        "random_minus_hot_partial_auc": float(
            np.mean(random_aucs) - hot_auc
        ),
    }


def save_cam(cam, path):
    if cam is not None:
        Image.fromarray(
            (cam * 255).astype(np.uint8)
        ).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="ReefEye XAI diagnostics"
    )

    parser.add_argument(
        "images",
        type=Path,
        help="Image file or folder",
    )

    parser.add_argument("--row", type=int)
    parser.add_argument("--column", type=int)

    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
    )

    parser.add_argument(
        "--random-trials",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--baseline",
        choices=["mean", "blur"],
        default="mean",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("xai_validation"),
    )

    args = parser.parse_args()

    if (args.row is None) != (args.column is None):
        parser.error(
            "Use --row and --column together"
        )

    if args.random_trials < 2 or args.max_images < 1:
        parser.error(
            "Use at least 2 random trials and 1 image"
        )

    if not args.images.exists():
        parser.error(
            "Image path does not exist"
        )

    if args.images.is_dir() and args.row is not None:
        parser.error(
            "Explicit coordinates require a single image file"
        )

    if args.images.is_file():
        files = [args.images]
    else:
        files = sorted(
            path
            for path in args.images.rglob("*")
            if path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }
        )

    files = files[:args.max_images]

    if not files:
        parser.error("No images found")

    model_dir = (
        Path(__file__).resolve().parent
        / "model"
    )

    config = json.loads(
        (model_dir / "config.json").read_text()
    )

    checkpoint = torch.load(
        model_dir / "reef_point_classifier.pt",
        map_location="cpu",
        weights_only=True,
    )

    if (
        config["architecture"] != "vit_b_16"
        or checkpoint["architecture"] != config["architecture"]
        or checkpoint["classes"] != config["classes"]
    ):
        raise ValueError(
            "Model and configuration do not match"
        )

    model = vit_b_16(
        weights=None,
        num_classes=len(config["classes"]),
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.eval()
    del checkpoint

    size = int(config["crop_size"])

    if size != model.image_size:
        raise ValueError(
            "Crop size does not match the ViT input size"
        )

    temperature = float(config["temperature"])

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Temperature must be finite and positive"
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            config["mean"],
            config["std"],
        ),
    ])

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    samples = []

    for index, path in enumerate(files):
        print(
            f"Trained model: {index + 1}/{len(files)}",
            flush=True,
        )

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

        row = (
            image.height // 2
            if args.row is None
            else args.row
        )

        column = (
            image.width // 2
            if args.column is None
            else args.column
        )

        left = column - size // 2
        top = row - size // 2

        if (
            left < 0
            or top < 0
            or left + size > image.width
            or top + size > image.height
        ):
            raise ValueError(
                f"A complete {size}px crop is required: {path}"
            )

        crop = image.crop((
            left,
            top,
            left + size,
            top + size,
        ))

        tensor = transform(crop).unsqueeze(0)

        with torch.inference_mode():
            target = int(
                model(tensor).argmax(dim=1)
            )

        _, cam = gradcam(
            model,
            tensor,
            target,
        )

        folder = (
            args.output
            / f"{index + 1:03d}_{path.stem}"
        )

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        crop.save(folder / "crop.png")
        save_cam(cam, folder / "trained_cam.png")

        record = {
            "image": str(path),
            "row": row,
            "column": column,
            "target_label": config["classes"][target],
            "trained_has_signal": cam is not None,
            "randomization": [],
            "deletion": None,
        }

        if cam is not None:
            if args.baseline == "mean":
                # Zero in normalized space = training RGB mean.
                baseline = torch.zeros_like(tensor)
            else:
                blurred = crop.filter(
                    ImageFilter.GaussianBlur(12)
                )

                baseline = transform(
                    blurred
                ).unsqueeze(0)

            record["deletion"] = deletion_test(
                model,
                tensor,
                baseline,
                cam,
                target,
                temperature,
                args.random_trials,
                1000 + index,
            )

        records.append(record)

        samples.append((
            tensor,
            target,
            cam,
            folder,
        ))

    for seed in args.seeds:
        print(
            f"Randomized model seed {seed}",
            flush=True,
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)

            random_model = vit_b_16(
                weights=None,
                num_classes=len(config["classes"]),
            )

            # Torchvision's default ViT head is zero-initialized.
            # A nonzero random head is needed for meaningful gradients.
            torch.nn.init.trunc_normal_(
                random_model.heads.head.weight,
                std=0.02,
            )

            torch.nn.init.zeros_(
                random_model.heads.head.bias
            )

        random_model.eval()

        for record, sample in zip(records, samples):
            tensor, target, cam, folder = sample

            # Keep the trained model's target class fixed.
            _, random_cam = gradcam(
                random_model,
                tensor,
                target,
            )

            metrics = compare_maps(
                cam,
                random_cam,
                size // model.patch_size,
            )

            metrics.update(
                seed=seed,
                random_has_signal=random_cam is not None,
            )

            record["randomization"].append(
                metrics
            )

            save_cam(
                random_cam,
                folder / f"random_cam_{seed}.png",
            )

        del random_model

    comparisons = [
        item
        for record in records
        for item in record["randomization"]
        if item["status"] == "compared"
    ]

    deletions = [
        record["deletion"]
        for record in records
        if record["deletion"] is not None
    ]

    summary = {
        "images": len(records),
        "valid_map_comparisons": len(comparisons),
        "inconclusive_comparisons": (
            len(records) * len(args.seeds)
            - len(comparisons)
        ),
        "nearly_identical_comparisons": sum(
            item["nearly_identical"]
            for item in comparisons
        ),
        "mean_pearson": (
            float(np.mean([
                item["pearson"]
                for item in comparisons
            ]))
            if comparisons
            else None
        ),
        "deletion_evaluated_crops": len(deletions),
        "mean_random_minus_hot_partial_auc": (
            float(np.mean([
                item["random_minus_hot_partial_auc"]
                for item in deletions
            ]))
            if deletions
            else None
        ),
    }

    report = {
        "protocol": {
            "baseline": args.baseline,
            "random_seeds": args.seeds,
            "random_trials": args.random_trials,
            "temperature": temperature,
            "target": "fixed trained-model predicted class",
            "selection": "explicit coordinates or image center",
            "deletion_unit": "ViT patch",
            "fractions_requested": FRACTIONS,
        },
        "summary": summary,
        "samples": records,
    }

    report_path = args.output / "report.json"

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        )
    )

    print(json.dumps(summary, indent=2))

    print(
        f"Report saved to {report_path.resolve()}"
    )

    print(
        "Diagnostics only: no automatic claim "
        "that explanations are valid."
    )


if __name__ == "__main__":
    main()