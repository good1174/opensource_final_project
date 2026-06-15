# Image Temporal Analysis Pipeline

An integrated pipeline for automatic temporal prediction and analysis of images. Leverages hidden states from vision-language models (Qwen v3.5) to learn and perform temporal predictions for extracting temporal information (month, day, hour, minute) from images.

## Pipeline Overview

This project consists of a fully automated 6-stage pipeline:

### 1 **make_dataset** - Dataset Generation
- Collects images from image folder
- Generates image descriptions using Gemini 2.5 Flash (ShareGPT4V format)
- Extracts image capture time from EXIF metadata
- Extracts GPS information (latitude/longitude)
- **Output**: `dataset.csv`

**Generated Columns:**
```
File Name, Description, DateTime, Full Path, Status, latitude, longitude, Is_Temporal
```

### 2 **embedding** - Image Embedding Extraction
- Uses Qwen 3.5 9B vision model
- Extracts hidden states from layers 9/19/24
- Saves as 4096-dimensional float16 tensors
- **Input**: `dataset.csv` + image folder
- **Output**: `{image_name}.jpg.pt` files (each containing `hidden_states9/19/24`)

### 3 **train** - Model Training for Temporal Prediction
- Loads image embeddings + temporal information
- Trains temporal prediction probes independently for 3 layers
- Applies ensemble weights
- **Input**: `.pt` embeddings + `dataset.csv`
- **Output**: `best_layer_{9/19/24}.pth` model weights + analysis CSV

### 4 **inference** - Inference on New Data
- Predicts temporal information for new images using trained weights
- Converts angles to time (2026 baseline)
- **Input**: new CSV + `.pt` embeddings + trained model
- **Output**: `inference_results.csv` (predicted time + error)

### 5 **train_eda** - Training Results Visualization
- Analyzes MAE, RMSE by layer
- Visualizes error distribution by time period
- Generates scatter plots / heatmaps
- **Input**: `individual_layer_{9}.csv`

### 6 **inference_eda** - Inference Results Visualization
- Compares predicted vs actual values
- Analyzes and visualizes error statistics
- **Input**: `inference_results.csv`

---

## Quick Start

### System Requirements

- **OS**: Windows / Linux / macOS
- **Python**: 3.8+
- **CUDA**: 11.8+ (GPU recommended)
- **conda environment**: `share2`

### Environment Setup

#### 1. Install packages in conda environment

```bash
conda activate share2
pip install -r requirements.txt
```

#### 2. GCP Service Account Setup (required for make_dataset stage)

Place the `gcp-key.json` file in the `forgithub/` folder.
- Must match your Google Cloud Project ID
- Requires Vertex AI / Gemini API permissions
- **For detailed instructions on creating a GCP project, enabling Vertex AI, and generating service account keys**, see:
  - [GCP Service Account Creation Guide (Google Documentation)](https://docs.cloud.google.com/iam/docs/service-accounts-create?hl=en)

---

## Usage

### Basic Execution

```bash
cd data_refines_time/forgithub
conda activate share2
python pipeline.py
```

### Control Execution by Stage

Edit the `STEPS` dictionary in `pipeline.py` to run only desired stages:

```python
STEPS = {
    "make_dataset":  True,      # Image → CSV + Descriptions
    "embedding":     True,      # Image → .pt embeddings
    "train":         True,      # Train model
    "inference":     False,     # Inference (optional)
    "train_eda":     False,     # Training results visualization (optional)
    "inference_eda": False,     # Inference results visualization (optional)
}
```

### Customize Paths

Modify the "Common Paths" section in `pipeline.py`:

```python
# 1. Common Paths
IMAGE_DIR       = r"C:\Users\...\images"        # ★ Original image folder
DATASET_CSV     = os.path.join(BASE_DIR, "dataset.csv")
PT_DIR          = r"D:\...\embedding_results"   # ★ .pt save location
MODEL_SAVE_DIR  = os.path.join(BASE_DIR, "saved_models")
RESULT_DIR      = os.path.join(BASE_DIR, "analysis_result")
GCP_KEY_PATH    = os.path.join(BASE_DIR, "gcp-key.json")

# 2. GCP Configuration
PROJECT_ID      = "your-project-id"
```

---

## Directory Structure

```
forgithub/
├── pipeline.py                      # ← Main pipeline (run this only!)
├── make_dataset.py                  # Dataset generation
├── make_dataset_embedding.py        # Image embedding extraction
├── train_code.py                    # Model training
├── inference_code.py                # Inference
├── train_eda.py                     # Training results visualization
├── inference_eda.py                 # Inference results visualization
├── requirements.txt                 # Package dependencies
├── gcp-key.json                     # GCP service account key
├── dataset.csv                      # Generated (step 1)
├── saved_models/                    # Generated (step 3)
│   ├── best_layer_9.pth
│   ├── best_layer_19.pth
│   └── best_layer_24.pth
└── analysis_result/                 # Generated (steps 3, 5, 6)
    ├── individual_layer_9.csv
    ├── individual_layer_19.csv
    ├── individual_layer_24.csv
    └── inference_results.csv
```

---

## Core Configuration Values

### Image Processing

| Configuration | Default | Description |
|---------------|---------|-------------|
| `DESC_MAX_SIDE` | 1024 px | Image longest side before description generation |
| `EMBED_IMAGE_SIDE` | 448 px | Image longest side before embedding extraction |
| `EMPTY_CACHE_EVERY` | 50 | Clear CUDA cache every N images |

### Model & Training

| Configuration | Default | Description |
|---------------|---------|-------------|
| `LAYERS` | [9, 19, 24] | Qwen hidden states layers to extract |
| `WEIGHTS` | {9: 0.34, 19: 0.33, 24: 0.33} | Ensemble weights |
| `EPOCHS` | 10 | Training epochs |
| `BATCH_SIZE` | 8 | Batch size |
| `LR` | 1e-4 | Learning rate |
| `YEAR` | 2026 | Baseline year for temporal conversion |

### Gemini API

| Configuration | Default | Description |
|---------------|---------|-------------|
| `GEMINI_MODEL` | "gemini-2.5-flash" | Description generation model |
| `DESC_TEMP` | 0.2 | Generation temperature (lower = more consistent) |
| `DESC_MAX_TOKENS` | 2048 | Maximum tokens to generate |

---

## Output Format

### dataset.csv Example
```
File Name,Description,DateTime,Full Path,Status,latitude,longitude,Is_Temporal
IMG001.jpg,"A photo of...",2025-06-15 14:30:00,C:\images\IMG001.jpg,1,37.5665,-122.4194,True
IMG002.jpg,"A photo of...",2025-06-15 15:45:00,C:\images\IMG002.jpg,1,37.5665,-122.4194,True
```

### inference_results.csv Example
```
File Name,Actual DateTime,Predicted DateTime,Error,Error_Format
IMG001.jpg,2025-06-15 14:30:00,2025-06-15 14:28:15,105,0:01:45
IMG002.jpg,2025-06-15 15:45:00,2025-06-15 15:44:50,10,0:00:10
```

---

## Troubleshooting

### CUDA Out of Memory
```python
EMPTY_CACHE_EVERY = 10  # Clear cache more frequently
BATCH_SIZE = 4          # Reduce batch size
```

### GCP Authentication & Vertex AI Errors

```bash
# Check GCP configuration
gcloud auth application-default login
# Or verify gcp-key.json path
```
- If authentication errors occur due to API activation or IAM permission issues, refer to the **[GCP Service Account Guide (Google Documentation)](https://docs.cloud.google.com/iam/docs/service-accounts-create?hl=en)**'s "Troubleshooting & Permission Recommendations" section to re-check your permissions.

---

## License

### Code License

**This project code is licensed under the [Apache License 2.0]

### Important: Two Usage Scenarios

#### Scenario 1: WITHOUT Using Pretrained Temporal Probe Model (Apache 2.0 License)

If you **train your own temporal prediction model from scratch** using:
- Your own images
- Qwen 3.5 9B embeddings
- Your custom training code

Then you can use and distribute this code under **Apache License 2.0** with no restrictions.

**To do this:**
- Only run the pipeline with `make_dataset: True` → `embedding: True` → `train: True`
- Do NOT use the pretrained model files
- Train your own model using the pipeline

#### Scenario 2: USING Pretrained Temporal Probe Model ( CC-BY-NC-4.0 Restriction)

If you use **the pretrained temporal probe model** (trained on ShareGPT4V dataset), then:

**License Restriction: CC-BY-NC-4.0** (Non-Commercial Only)
- Commercial use is **NOT permitted**
- Non-commercial research/educational use is permitted
- Must provide copyright attribution

** Download Pretrained Model:**

The pretrained temporal probe model is available at:
```
https://drive.google.com/file/d/1ujluIUUaw_n3hsLLzCBuVEJZ9ATCIKRc/view?usp=sharing
```

Extract the `.pth` file and place in `saved_models/` folder:
```
forgithub/saved_models/
├── best_layer_9.pth       ← Download and place here
├── best_layer_19.pth      ← Download and place here
└── best_layer_24.pth      ← Download and place here
```

Then use the pipeline with:
```python
STEPS = {
    "make_dataset":  False,
    "embedding":     True,   # Extract embeddings only
    "train":         False,  # Skip training
    "inference":     True,   # Use pretrained model for inference
    ...
}
```

### Dependency Licenses

| Component | License | Description |
|-----------|---------|-------------|
| **Qwen 3.5 9B** | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Vision-language model for image embedding extraction |
| **ShareGPT4V Dataset** | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Image description generation reference dataset (only if using pretrained model) |


### Copyright Attribution

If using the pretrained model or ShareGPT4V dataset:
```
This project references the data format and caption generation methodology of ShareGPT4V (Lin-Chen et al., 2023).
ShareGPT4V Dataset: https://huggingface.co/datasets/Lin-Chen/ShareGPT4V
```

---

## Contact

For questions or issues regarding this project, please contact me at good117454@gmail.com.
