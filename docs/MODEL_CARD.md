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

## Test results

ViT-B/16 was selected from three candidate architectures using validation Macro F1, achieving **0.7487** on the validation set. Its final evaluation uses six held-out dive sites, with separate sites used for training, validation, and confidence calibration.

### Classification performance

Both test views use the same six held-out sites and the seven retained labels, with points sampled inside image boundaries. The per-class view samples up to two points per class per image; the uniform view samples up to six eligible points per image.

| Metric | Per-class sampling (`test_balanced`) | Uniform eligible-point sampling (`test_uniform`) |
| --- | ---: | ---: |
| Evaluated points | 3,272 | 3,148 |
| Accuracy | 77.72% | **84.82%** |
| Macro F1 | **0.7525** | 0.6493 |
| Weighted F1 | 0.7781 | **0.8668** |
| Balanced accuracy | 74.06% | 76.40% |
| ECE before calibration | 0.1263 | 0.0744 |
| ECE after calibration | **0.0362** | **0.0184** |

Temperature scaling improved confidence calibration in both test views. ECE uses 15 confidence bins, with lower values indicating closer agreement between confidence and observed accuracy.

### Confidence-guided review

At the calibrated confidence threshold of approximately **67.82%**, the model achieved **91.61% accuracy among above-threshold predictions** in the uniform test view. These predictions cover **81.4%** of evaluated points, while **18.6%** are flagged for human review.

| Metric | Per-class sampling | Uniform eligible-point sampling |
| --- | ---: | ---: |
| Accuracy among above-threshold predictions | 86.04% | **91.61%** |
| Above-threshold coverage | 76.7% | **81.4%** |
| Points flagged for review | 23.3% | 18.6% |

The above-threshold accuracy is measured before manual correction, on the subset meeting the confidence threshold.

Results are transcribed from the supplied notebook's saved evaluation outputs for exported run `b03f3ddae1699f26`, matching the `run_key` prefix in [config.json](../backend/model/config.json). Percentages and scores are rounded for display.

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
