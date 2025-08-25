# 🏋️ Training Methods

This folder contains the **training scripts** developed for the thesis.  
The focus is on different training strategies, including the proposed **Two-Stream architecture**.

---

## 📂 Included Scripts

- `model_training.py` → Standard training loop with early stopping and logging utilities.  
- `two_streams_training.py` → Training procedure for the **Two-Stream model**, where two separate input channels are processed in parallel.  
- `DINO_fine_tuning.py` → Script to fine-tune the DINO model on the chosen dataset.  

---

## 📌 Notes
 
- All scripts include utilities such as early stopping, logging of loss/accuracy, and optional checkpoint saving.  