# Automatic Nano-Particle Detector

A deep‑learning pipeline for **particle segmentation and counting** in grayscale images using a U‑Net model with **sliding‑window** training/inference and robust post‑processing.

- Trains on large images via patch extraction with Albumentations.
- Predicts full‑resolution masks with stitched sliding‑window inference.
- Counts/filters instances, measures size/shape stats, and exports CSV/JSON + visual overlays.

---

## Directory Structure
```
particle-segmentation/
│
├── data/
│   ├── cropped_images/               # training images (.tif by default)
│   └── cropped_masks/                # binary masks named `<basename>_mask.png`
│
├── scripts/
│   ├── crop_bottom.py               # 
│   ├── csv_to_mask.py   
│   └── utils.py
│
├── experiment/               # training runs & checkpoints (auto‑created)
├── predictions/              # inference outputs (auto‑created)
│
├── dataset.py                # SlidingWindowSegmentationDataset (patches + aug)
├── model.py                  # U‑Net (pretrained encoder)
├── train.py                  # training loop (schedulable mask dilation→erosion)
├── predict.py                # inference + post‑processing + measurement
├── utils.py                  # helpers (seed, metrics, etc.)
├── requirements.txt          # dependencies
├── visual.py                 # visualize the predictions
└── README.md                 # you’re here
```

---

## Getting Started

### 1) Install Dependencies
We recommend a fresh virtual environment.

```bash
git clone https://github.com/prashansapkota/particle-segmentation.git
cd particle-segmentation

# Create env (example with conda)
conda create -n particleseg python=3.10 -y
conda activate particleseg

# Install PyTorch (pick the right CUDA build from pytorch.org)
# Example for CUDA 12.1:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Project deps
pip install -r requirements.txt
```

### 2) Prepare Your Data
Place **grayscale** training images under `data/cropped_images/` and their **binary masks** under `data/masks/`.

- Masks should follow the naming convention: `image123.tif` ↔ `image123_mask.png`  
- If a basename already ends with `_mask`, the code accepts `<basename>.png`.

```
data/
  cropped_images/
    sample_001.tif
    sample_002.tif
  cropped_masks/
    sample_001_mask.png
    sample_002_mask.png
```

> **Tip:** Large images are handled via patching; no tiling needed from you.

---

## 🏋️ Training
Run training with your desired experiment folder. (Use `wandb` to log if desired.)

```bash
python main.py   --images_dir data/images   --masks_dir  data/masks   --epochs 2000   --batch_size 8   --lr 1e-3   --experiment_name experiment/run_$(date +%Y%m%d_%H%M%S)   --val_split 0.2   --dilation_iters 10   --erosion_freq 150   --erosion_iters 1   --seed 42   [--wandb]
```

During training the code can **start with dilated masks** (more forgiving) and gradually **erode** toward the original labels to increase difficulty.

**Defaults & design choices**
- Patch size **256**, stride **128** for both training & inference.
- Images are normalized with ImageNet statistics and grayscale is **replicated to 3 channels**.
- Dice is reported on binarized predictions.
- Checkpoints: `best_model.pth`, periodic `model_epoch_XXXX.pth` (every N epochs), and `Final_Model.pth` saved under `--experiment_name`.

### 🧩 Argument Reference (Training)

| Argument | Description | Default |
|---|---|---|
| `--images_dir` | Directory with training images | `data/images` |
| `--masks_dir` | Directory with binary masks | `data/masks` |
| `--epochs` | Number of epochs | `2000` |
| `--batch_size` | Batch size | `8` |
| `--lr` | Learning rate | `1e-3` |
| `--experiment_name` | Output folder for run/checkpoints | `experiment/run_YYYYMMDD_HHMMSS` |
| `--val_split` | Validation fraction | `0.2` |
| `--dilation_iters` | Initial dilation on masks | `10` |
| `--erosion_freq` | Erode every N epochs | `150` |
| `--erosion_iters` | Erosion steps each time | `1` |
| `--seed` | Random seed | `42` |
| `--wandb` | Enable Weights & Biases logging | off |

> **Loss:** The code is compatible with `BCEWithLogitsLoss` (optionally with `pos_weight`) or a custom focal+Dice combination. Ensure your model returns **logits** (no final sigmoid in the forward pass when using BCE with logits).

---

## 🔮 Inference (Prediction)
Run sliding‑window inference on a single image and export masks, labels, overlays, and instance measurements.

```bash
python predict.py   --image data/new_images/sample.tif   --run_dir experiment/run_20250805_104636   --out_dir predictions   --thresh 0.25   --min_size 5   --morph_ksize 3   --watershed   --min_distance 5   --min_circularity 0.3   --patch_size 256   --stride 128   --crop_bottom 100
```

**Checkpoint policy:** The script picks the **latest** `model_epoch_*.pth` in `--run_dir`. If none, it falls back to `Final_Model.pth`. It **does not** use `best_model.pth`.

### Output Layout (per‑image subfolder)
```
predictions/
  sample/                  # created automatically for `sample.tif`
    prob.png               # probability map
    bin.png                # threshold + morph open + small‑object removal
    labels.png             # color labels over image
    overlay.png            # predicted centers overlaid on grayscale
    particles.csv          # label_id, x, y, area, equiv_diameter, circularity
    summary.json           # counts + summary stats + the parameters used
    distance_hist.png      # (optional) inter‑particle distance histogram
```

> **Note:** High histogram counts occur if you plot **all pairwise distances** (O(n²)); prefer nearest‑neighbor distances for a 1‑per‑particle view.

---

## 🧠 Model & Channels
- U‑Net with a pretrained encoder.
- Dataset replicates grayscale → **3‑channel** and applies ImageNet normalization.
- Ensure `model.py` `in_channels` matches your dataset output (set to **3** to use ImageNet weights end‑to‑end).

---

## 🧪 Tips & Troubleshooting
- **Mask naming**: `*_mask.png` must match image basenames.
- **Missing checkpoints**: Verify `--run_dir`; the script searches for `model_epoch_*.pth` first.
- **OOM on large images**: Reduce `--patch_size` or increase `--stride`.
- **Loss stability**: Use `nn.BCEWithLogitsLoss` and pass `pos_weight = n_neg/n_pos` for class imbalance.
- **Determinism**: `utils.set_seed` controls RNG and cuDNN flags.

---

## 📜 Citation
If this project helps your work, please cite this repository and the libraries it builds on (PyTorch, Albumentations, scikit‑image).

```
@software{particle_segmentation_2025,
  title        = {Automatic Nano-Particle Detector},
  author       = {Prashan Sapkota},
  year         = {2025},
  url          = {https://github.com/prashansapkota/particle-segmentation}
}
```

---

## 📄 License
Specify your license (e.g., MIT) in `LICENSE`.
