# 🖨️ Output Handlers

This folder contains helper modules for **managing and saving training outputs** such as logs, metrics, and visualizations.
They are **not standalone executables**, but are imported and used during the training pipeline.

---

## 📂 Included Modules

- `loss_acc_graph.py` → Functions to plot and save training/validation **loss** and **accuracy** curves.
- `my_confusion_matrix.py` → Utility to generate and save a **confusion matrix** from predictions.
- `out_logger_creation.py` → Creates loggers for saving training logs (e.g., epoch accuracy).
- `output_dir_creation.py` → Creates structured output directories (`runs/runX/`) for each training session.
