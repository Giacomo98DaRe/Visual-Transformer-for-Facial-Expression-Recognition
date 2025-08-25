# 🎓 Master Thesis — Model Implementation

This folder contains the **core implementation** of my Master's Thesis project.  
It includes data handling, model definitions, training methods, utilities, and the main entry point script `model_implementation.py`.

⚠️ **Note:** Some relative paths may be incorrect due to the recent refactor of the code. Adjust them if needed when running the scripts.

---

## 🚀 Entry Point

- **`model_implementation.py`** → Main script of the thesis.  
  - Defines random seeds and device setup (CPU/GPU).  
  - Loads and instantiates models (ViT, Swin, Poster, Dino + Poster).  
  - Supports **One-Stream** and **Two-Stream** training strategies.  
  - Handles dataset creation, weighted samplers, dataloaders, and logging.  
  - Calls the appropriate training method and saves results (loss, accuracy, confusion matrix).

---

## 🗂 Folder Structure

```
📁 data/               # Dataset samples and paths (local and full)
📁 dataset_handlers/   # Custom dataset classes (AffectNet, Two-Stream)
📁 docker/             # Dockerfile and requirements for reproducible GPU runs
📁 models/             # Model definitions (ViT, Swin, Dino, Poster, custom)
📁 output_handlers/    # Utilities for logging, plots, confusion matrices
📁 runs/               # Outputs of training runs (logs, checkpoints, results)
📁 tools/              # Utility scripts (sampling, plotting, thesis helpers)
📁 training_methods/   # Training scripts (standard, two-stream, fine-tuning)
📄 model_implementation.py   # Main entry point
📄 README.md
```

---


## 🏋️ Training Strategies

- **Standard training** (single input stream).  
- **Two-Stream training** → core thesis contribution:  
  - Processes image landmarks and extracted features in **parallel streams**.  
  - Improves emotion recognition by combining heterogeneous representations.  

---

## 📌 Notes

- Select the different folders for further details.
