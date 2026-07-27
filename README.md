<<<<<<< HEAD
# Facial Expression Recognition using CNN and MediaPipe Landmark Heatmaps

This project implements facial expression recognition on the FER-2013 dataset using a baseline CNN model and a proposed landmark-guided CNN extension. The baseline follows `Proposed_Model_2` from the reference paper, while the extension adds MediaPipe facial landmark heatmaps as a second input channel.

## Project Overview

Facial Expression Recognition (FER) is the task of classifying a face image into an emotion category. This project uses the FER-2013 dataset, where each image is a `48 x 48` grayscale face image.

The seven emotion classes are:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

## Main Idea

The baseline CNN learns expression features only from grayscale pixel appearance. However, facial expressions are also strongly related to facial geometry, such as eyebrow shape, eye opening, mouth movement, and face boundary changes.

To include this geometry information, the proposed extension detects facial landmarks using MediaPipe and converts them into Gaussian heatmaps. The grayscale image and heatmap are then combined using early fusion before being passed into the CNN.

## Methodology

### 1. Baseline Model

The baseline model is based on `Proposed_Model_2` from the reference paper.

Baseline input:

```text
1 x 48 x 48 grayscale image
```

Baseline pipeline:

```text
FER-2013 grayscale image -> Proposed_Model_2 CNN -> Dense layer -> Softmax -> Emotion class
```

The baseline uses only the original grayscale image. It does not use MediaPipe, landmarks, heatmaps, or super-resolution.

### 2. Proposed Extension: LM-CNN

The proposed extension is a Landmark-Guided CNN (LM-CNN).

Extension pipeline:

```text
48 x 48 grayscale image
-> create upscaled copy using EDSR_x4 / resize
-> MediaPipe detects 468 facial landmarks
-> scale landmark coordinates back to 48 x 48
-> create Gaussian landmark heatmap
-> concatenate grayscale image + heatmap
-> LM-CNN classifier
-> predicted emotion class
```

Important point:

```text
Super-resolution is used only for landmark detection.
The CNN still receives 48 x 48 input.
```

### 3. Early Fusion

Early fusion means the original image and landmark heatmap are combined before entering the CNN.

```text
Grayscale image: 1 x 48 x 48
Heatmap:         1 x 48 x 48
Fused input:     2 x 48 x 48
```

The grayscale channel gives appearance information, while the heatmap channel gives facial-geometry information.

## Folder Structure

Expected project structure:

```text
Face_recognition/
├── data/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   ├── test/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── processed/
│       └── heatmaps/
├── models/
│   └── EDSR_x4.pb
├── checkpoints/
├── outputs/
├── scripts/
│   ├── precompute_heatmaps.py
│   ├── test_dataset.py
│   ├── test_landmarks.py
│   └── test_model.py
├── src/
│   └── fer_cnn/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── heatmaps.py
│       ├── landmarks.py
│       ├── model.py
│       ├── train.py
│       └── train_baseline.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Environment Setup

This project uses `uv` for Python environment and dependency management.

Install dependencies:

```bash
uv sync
```

Run commands inside the project folder:

```bash
cd /Users/emilyjoy/Documents/Face_recognition
```

## Dataset

The dataset should be arranged as an ImageFolder-style dataset:

```text
data/train/<emotion_name>/<image_files>
data/test/<emotion_name>/<image_files>
```

Example:

```text
data/train/happy/Training_12345.jpg
data/test/sad/PublicTest_12345.jpg
```

### Where to Get the Dataset

FER-2013 can be downloaded from Kaggle:

- FER-2013 ImageFolder version: https://www.kaggle.com/datasets/msambare/fer2013
- Alternative FER-2013 folder version: https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013

After downloading, place the dataset in this structure:

```text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Do not upload the full FER-2013 dataset to GitHub. Keep it local because datasets can be large and may have distribution or licensing restrictions.

## Super-Resolution Model

This project optionally uses the OpenCV super-resolution model `EDSR_x4.pb` to improve MediaPipe landmark detection on small `48 x 48` FER-2013 images.

The model can be obtained from OpenCV's super-resolution model resources:

- OpenCV `dnn_superres` documentation: https://docs.opencv.org/master/d5/d29/tutorial_dnn_superres_upscale_image_single.html
- `EDSR_x4.pb` model file: https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb

Place the downloaded file here:

```text
models/EDSR_x4.pb
```

Important:

```text
EDSR_x4 is used only to create a clearer image copy for MediaPipe landmark detection.
The CNN itself still trains on 48 x 48 input.
```

If `models/EDSR_x4.pb` is not available, the code can still use normal high-quality image resizing as a fallback for landmark detection.

## Heatmap Preprocessing

Before training the LM-CNN extension, precompute landmark heatmaps:

```bash
uv run python scripts/precompute_heatmaps.py
```

This creates heatmap files under:

```text
data/processed/heatmaps/
```

Each heatmap corresponds to one original FER-2013 image.

## Testing Individual Parts

Test MediaPipe landmark detection and heatmap generation:

```bash
uv run python scripts/test_landmarks.py
```

Test dataset loading:

```bash
uv run python scripts/test_dataset.py
```

Expected dataset batch shape for LM-CNN:

```text
Image batch shape: torch.Size([4, 2, 48, 48])
Label batch shape: torch.Size([4])
```

Test model input and output:

```bash
uv run python scripts/test_model.py
```

Expected model output:

```text
Input shape: torch.Size([4, 2, 48, 48])
Output shape: torch.Size([4, 7])
```

## Training

Train the proposed LM-CNN extension:

```bash
uv run python src/fer_cnn/train.py
```

Train the baseline Proposed_Model_2 model:

```bash
uv run python src/fer_cnn/train_baseline.py
```

## Results

Latest recorded results:

| Model | Input | Accuracy | Macro F1 |
|---|---|---:|---:|
| Proposed_Model_2 Baseline | Grayscale only | 61.21% | 59.76% |
| LM-CNN Extension | Grayscale + heatmap | 65.17% | 63.20% |
| Improvement | Appearance + geometry | +3.96 pp | +3.44 pp |

The LM-CNN extension improved test accuracy compared with the grayscale-only baseline.

## Why the Extension Can Help

The baseline CNN learns appearance features from grayscale pixels. The landmark heatmap adds facial-geometry information from important facial regions such as the eyes, eyebrows, nose, mouth, and face boundary.

In short:

```text
Grayscale image = appearance
Landmark heatmap = facial geometry
Early fusion = appearance + geometry
```

This gives the CNN more information about expression-related shape changes.

## Reference

M. K. Rusia and D. K. Singh, "An efficient CNN approach for facial expression recognition with some measures of overfitting," 2021. DOI: `10.1007/s41870-021-00803-x`
=======
