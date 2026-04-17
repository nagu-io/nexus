@echo off
chcp 65001>nul
echo ==============================================
echo NEXUS LLM TRAINING PIPELINE v3 (FIXED)
echo ==============================================

echo [0/6] Setting up D: drive environment...
mkdir D:\nexus\.nexus_pkgs 2>nul
mkdir D:\temp 2>nul
mkdir D:\pip_cache 2>nul
mkdir D:\hf_cache 2>nul
mkdir D:\hf_cache\hub 2>nul

set PYTHONPATH=D:\nexus\.nexus_pkgs
set PIP_USER=0
set TMP=D:\temp
set TEMP=D:\temp
set PIP_CACHE_DIR=D:\pip_cache
set HF_HOME=D:\hf_cache
set HF_HUB_CACHE=D:\hf_cache\hub
set TRANSFORMERS_CACHE=D:\hf_cache\hub
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
set UNSLOTH_LLAMA_CPP_PATH=D:\unsloth\llama.cpp

echo [1/6] Installing Framework Stack (pinned compatible versions)...
REM py -3.11 -m pip install --target D:\nexus\.nexus_pkgs --no-user ^
REM    torch==2.5.1 torchvision==0.20.1 ^
REM    --index-url https://download.pytorch.org/whl/cu124 ^
REM    --extra-index-url https://pypi.org/simple ^
REM    "torchao==0.12.0" ^
REM    "transformers>=4.51.3,<=5.5.0" ^
REM    "trl>=0.18.2,<=0.24.0" ^
REM    "datasets>=3.4.1,<4.4.0" ^
REM    "peft>=0.18.0" ^
REM    "accelerate>=0.34.1" ^
REM    "bitsandbytes>=0.45.5" ^
REM    "unsloth" ^
REM    "triton-windows==3.1.0.post17" ^
REM    "fsspec==2025.9.0" ^
REM    "rich" ^
REM    "sentencepiece>=0.2.0" ^
REM    --isolated

echo [2/6] Generating curated Nexus training dataset...
py -3.11 nexus\train\dataset_generator.py

echo [3/6] Verifying CUDA acceleration...
py -3.11 -c "import sys; sys.path.insert(0,r'D:\nexus\.nexus_pkgs'); import torch; assert torch.cuda.is_available(), 'NO CUDA'; print(f'Torch {torch.__version__} on {torch.cuda.get_device_name(0)}')"

echo [4/6] Starting VRAM-Optimized Fine-Tuning (6 epochs)...
py -3.11 nexus/train/unsloth_trainer.py

echo [5/6] Exporting to GGUF and registering with Ollama...
py -3.11 nexus/train/ollama_export.py

echo ==============================================
echo TRAINING COMPLETE. NEXUS-TRAINED IS ONLINE.
echo ==============================================
