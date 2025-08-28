# 🐍 Python Environment Setup

This folder contains Jupyter notebooks for training and evaluating BERT-based classifiers. To run them locally, you will need to set up a Python virtual environment and Jupyter kernel.

## 🔧 Setup Instructions

### 1. Install Python

Install Python 3.12.10 (recommended). You can download it from the [official Python website](www.python.org/downloads/).

### 2. Create and activate a virtual environment

In your terminal or command prompt:

```
# Navigate to the repository folder (example path shown below)
cd "C:\Users\YourName\Documents\GitHub\nonsig-master-thesis"

# Create a virtual environment
python -m venv nonsig_venv

# Activate the environment (Windows)
nonsig_venv\Scripts\activate

# (Mac/Linux)
source nonsig_venv/bin/activate
```

### 3. Install Jupyter kernel support

Inside the activated environment (still in your terminal):

```
pip install ipykernel
python -m ipykernel install --user --name nonsig_kernel
```

### Open and run notebooks in Visual Studio Code

1. Install [Visual Studio Code](https://code.visualstudio.com/).

2. Install the Python and Jupyter extensions from the VS Code Extensions Marketplace.

3. Open the repository folder (nonsig-master-thesis) in VS Code.

4. Open one of the notebooks in notebooks/python/.

5. In the top-right of the windwo, first click 'Select Kernal'. Then choose 'Jupyter Kernel...' and finally select the kernel you created earlier: 'nonsig_kernel'.

→ You can now run the notebooks cell by cell inside VS Code.

---

## ⚡ GPU Acceleration (Recommended)

If you have an NVIDIA GPU, you can speed up model training by installing CUDA 12.9:

👉 *[Download CUDA 12.9 here](https://developer.nvidia.com/cuda-12-9-0-download-archive)*

- Make sure your GPU is compatible.

- Install the correct CUDA toolkit version for your system.

- If no GPU is available, training will still work, but significantly slower.

