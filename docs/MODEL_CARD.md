# ReefEye model card

## Overview

ReefEye's released model classifies local crops around sampled image points. It provides a predicted label and a temperature-scaled confidence score, with lower-confidence points routed to human review.

This card documents the configuration and inference implementation shipped in this repository. Source files: [config.json](../backend/model/config.json) and [main.py](../backend/main.py).

| Field | Value |
| --- | --- |
| Architecture | `vit_b_16` — Vision Transformer, ViT-B/16 |
| Task recorded in configuration | Semantic-mask-derived point crop classification |
| Input | 224 × 224 RGB crop |
| Output | One of seven labels, confidence score, and review flag |
| Execution | CPU; evaluation mode with PyTorch inference mode |
| Checkpoint | [reef_point_classifier.pt — Model v1](https://github.com/Maram1alzahrani/ReefEye/releases/tag/model-v1) |

## Data provenance

The released configuration records the source dataset as [`josauder/coralscapesV2`](https://huggingface.co/datasets/josauder/coralscapesV2). Its upstream dataset card describes an image segmentation dataset. ReefEye's recorded task uses labels derived from semantic masks to classify point-centered image crops.

ReefEye applies crop classification at sampled image locations using the seven-label space and site split recorded in the released configuration.

## Label space

The checkpoint must use this exact ordered class list:

| Index | Stored label | Display name | Category |
| --- | --- | --- | --- |
| 0 | `acropora` | Acropora | Coral genus |
| 1 | `background` | Background | Contextual class |
| 2 | `galaxea` | Galaxea | Coral genus |
| 3 | `other_coral` | Other Coral | Grouped class |
| 4 | `pocillopora` | Pocillopora | Coral genus |
| 5 | `porites` | Porites | Coral genus |
| 6 | `stylophora` | Stylophora | Coral genus |

The retained label space combines named coral genera with Background and Other Coral categories.

## Inference pipeline

1. Read and validate the uploaded image.
2. Apply EXIF orientation and convert to RGB.
3. Generate evenly spaced row and column locations, keeping each crop fully inside the image.
4. Extract a 224 × 224 crop around each location.
5. Convert the crop to a tensor and normalize each RGB channel.
6. Run crops through the classifier in batches of up to eight.
7. Divide logits by the configured temperature and apply softmax.
8. Return the highest-scoring label and flag points below the review threshold.

| Setting | Value |
| --- | --- |
| RGB mean | `[0.485, 0.456, 0.406]` |
| RGB standard deviation | `[0.229, 0.224, 0.225]` |
| Temperature | `1.7974332571029663` |
| Review threshold | `0.6782287788391114` |
| Review rule | `confidence < review_threshold` |
| Coordinate format | `(row, column) = (y, x)`, in pixels after EXIF correction |

Grid locations are deduplicated to keep sampled coordinates unique.

## Evaluation configuration

The following evaluation metadata is recorded in the released configuration:

| Item | Recorded value |
| --- | --- |
| Split unit | Dive site |
| Training sites | 18 |
| Validation sites | 9 |
| Calibration sites | 8 |
| Test sites | 6 |
| Random seed | 42 |
| Evaluation scope | Retained eligible labels, inner-image points only |
| Calibration status | `target_met_on_calibration` |

The configuration also retains a `run_key` and `sampling_hash` to identify the recorded experiment and sampling setup.

## Intended use

- Explore predicted labels at sampled locations in reef photographs.
- Inspect the image context behind individual classifications.
- Prioritize uncertain points for expert review.
- Export original predictions alongside confirmed human labels.

Human review decisions are saved in the browser workspace alongside the original model predictions and confidence scores.

## Release integrity

The released checkpoint is 343,268,019 bytes (approximately 327 MiB).

SHA-256:

```text
d865da5662ecd676904e46cdfb0f8e211c24e8bcb8384c2b31e83754a42a0319
```

The backend loads the checkpoint on CPU with `weights_only=True`, verifies its architecture and ordered class list against the configuration, and loads the model state strictly.
