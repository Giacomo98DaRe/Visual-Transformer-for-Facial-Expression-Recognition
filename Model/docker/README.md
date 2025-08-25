# 🐳 Environment Setup (Docker + Conda)

This folder contains the configuration files needed to **set up and reproduce the training environment**. Models were originally trained using GPU resources provided by the university.

---

## 📦 Conda Environment

All dependencies are listed in the `requirements.yml` file.  
You can recreate the environment with:

```bash
conda env create -f requirements.yml
conda activate project-env
```

*(replace `project-env` with the environment name defined in the `.yml` file)*

---

## 🐳 Docker Setup

A `Dockerfile` is provided to run the training inside a container.  
This was mainly used to leverage the **university GPU cluster**.

### Build the image
```bash
docker build -t project-image .

### Run with GPU support
If GPUs are available (with NVIDIA runtime installed):

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace project-image
```

---

## ⚙️ Run Configurations

The `.runconfigs/` folder contains example run configurations and various usage notes..  

---

## ⚠️ Notes

- Use Conda locally for debugging and development.  
- Use Docker when training on GPU clusters, to ensure reproducibility.  
