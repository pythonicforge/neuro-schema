# Installation Guide

### System Requirements
- OS: Linux (Ubuntu 22.04 / 24.04 recommended)
- GPU: NVIDIA GPU with CUDA support (RTX 30XX or better)
- VRAM: 4GB minimum (6GB+ recommended)
- RAM: 16GB recommended
- Python: 3.10 (via Miniconda)
- CMake: ≥ 3.22

<br/>

### Environment Setup

1. Clone the repository
```bash
git clone https://github.com/pythonicforge/neuro-schema.git
cd neuro-schema
```
2. Create a new conda environment
```bash
conda create -n neuro-schema python=3.10 -y
conda activate neuro-schema
```
3. Install Python dependencies
```bash
pip install -r requirements.txt
```
4. Install essential build tools and CUDA-related libraries
```bash
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  libgomp1 \
  libomp-dev \
  curl \
  git \
  ninja-build
```
5. Clone llama.cpp locally
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUBLAS=ON
cmake --build . --config Release
```

**Note this file yet needs to be updated**
