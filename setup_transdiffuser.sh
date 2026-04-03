#!/bin/bash
# Complete Transdiffuser Project Setup for l4_env
# Updated: Aligned for RTX 5050 (sm_120), CUDA 13.0, Driver 580
# Navsim: handled via editable install (pip install -e), no PYTHONPATH needed
# Run this ONCE to make the entire project importable from anywhere

echo "=========================================="
echo "Transdiffuser Project Setup"
echo "=========================================="
echo ""

# Check current environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "  No conda environment active"
    echo "Please run: conda activate l4_env"
    exit 1
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo ""

# Project root - the main directory containing everything
PROJECT_ROOT="/home/phamtamadas/DPJI/transdiffuser"
DDPM_ROOT="$PROJECT_ROOT/DDPM"

echo "Project root: $PROJECT_ROOT"
echo "DDPM root: $DDPM_ROOT"
echo ""

# Verify project structure
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "  Project root not found at: $PROJECT_ROOT"
    exit 1
fi

if [ ! -d "$DDPM_ROOT" ]; then
    echo "  DDPM directory not found at: $DDPM_ROOT"
    exit 1
fi

echo "  Project structure found"
echo ""

# Ensure all directories have __init__.py files
echo "=========================================="
echo "Creating __init__.py files"
echo "=========================================="
echo ""

touch "$DDPM_ROOT/__init__.py"
touch "$DDPM_ROOT/encode/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/adapters/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/datasets/__init__.py" 2>/dev/null || true

# Note: navsim __init__.py files are managed by navsim's own package structure
# Do NOT touch navsim internals here - it's an editable install

echo "  Created __init__.py files"
echo ""

# Setup conda environment activation
echo "=========================================="
echo "Setting up conda environment hooks"
echo "=========================================="
echo ""

CONDA_ENV_DIR="$CONDA_PREFIX/etc/conda/activate.d"
CONDA_DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
ACTIVATE_SCRIPT="$CONDA_ENV_DIR/transdiffuser_setup.sh"
DEACTIVATE_SCRIPT="$CONDA_DEACTIVATE_DIR/transdiffuser_setup.sh"

mkdir -p "$CONDA_ENV_DIR"
mkdir -p "$CONDA_DEACTIVATE_DIR"

# Create activation script
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Transdiffuser Environment Setup
# Aligned for: RTX 5050 (sm_120), CUDA 13.0, Driver 580

# ---- Project Paths ----
export PROJECT_ROOT="/home/phamtamadas/DPJI/transdiffuser"
export DDPM_ROOT="$PROJECT_ROOT/DDPM"
export TRANSDIFFUSER_HOME="$DDPM_ROOT"

# ---- CUDA 13.0 Configuration ----
# Required for building CUDA extensions (DFA3D, etc.)
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Target architecture for RTX 5050 Laptop (Blackwell, sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"

# ---- PYTHONPATH ----
# Only DDPM_ROOT needed for: from encode.X, from adapters.X, from datasets.X
# Navsim is NOT added here - it's an editable pip install (pip install -e)
export PYTHONPATH="$DDPM_ROOT:$PYTHONPATH"

# ---- NAVSIM Data Paths (read-only references) ----
export NAVSIM_DEVKIT_ROOT="$DDPM_ROOT/datasets/navsim"
export OPENSCENE_DATA_ROOT="$DDPM_ROOT/datasets/navsim/download"
export NUPLAN_MAPS_ROOT="$DDPM_ROOT/datasets/navsim/download/maps"
EOF

chmod +x "$ACTIVATE_SCRIPT"

# Create deactivation script
cat > "$DEACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Clean up Transdiffuser environment

# Remove project vars
unset PROJECT_ROOT
unset DDPM_ROOT
unset TRANSDIFFUSER_HOME

# Remove CUDA overrides
# Note: only removes the paths we added, preserves system defaults
if [[ -n "$PATH" ]]; then
    export PATH=$(echo "$PATH" | sed 's|/usr/local/cuda-13.0/bin:||g')
fi
if [[ -n "$LD_LIBRARY_PATH" ]]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's|/usr/local/cuda-13.0/lib64:||g')
fi
unset CUDA_HOME
unset TORCH_CUDA_ARCH_LIST

# Remove DDPM from PYTHONPATH
if [[ -n "$PYTHONPATH" ]]; then
    DDPM_PATH="/home/phamtamadas/DPJI/transdiffuser/DDPM"
    export PYTHONPATH=$(echo "$PYTHONPATH" | sed "s|$DDPM_PATH:||g" | sed "s|:$DDPM_PATH||g" | sed "s|^$DDPM_PATH$||g")
fi

# Remove NAVSIM data path vars
unset NAVSIM_DEVKIT_ROOT
unset OPENSCENE_DATA_ROOT
unset NUPLAN_MAPS_ROOT
EOF

chmod +x "$DEACTIVATE_SCRIPT"

echo "  Created conda environment hooks"
echo ""

# Setup .pth file - only DDPM_ROOT (navsim handled by editable install)
echo "=========================================="
echo "Setting up .pth file"
echo "=========================================="
echo ""

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
PTH_FILE="$SITE_PACKAGES/transdiffuser.pth"

cat > "$PTH_FILE" << EOF
$DDPM_ROOT
EOF

echo "  Created .pth file: $PTH_FILE"
echo "  Content:"
cat "$PTH_FILE" | sed 's/^/    /'
echo ""

# Apply environment immediately
echo "=========================================="
echo "Applying environment"
echo "=========================================="
echo ""

source "$ACTIVATE_SCRIPT"

echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "nvcc: $(nvcc --version 2>&1 | grep release)"
echo ""
echo "PYTHONPATH (first 5 entries):"
echo "$PYTHONPATH" | tr ':' '\n' | head -5
echo ""

# Verification
echo "=========================================="
echo "Verifying Setup"
echo "=========================================="
echo ""

python << 'PYEOF'
import sys

print("=== Python & CUDA ===")
import torch
print(f"  PyTorch:    {torch.__version__}")
print(f"  CUDA:       {torch.version.cuda}")
arch_list = torch.cuda.get_arch_list()
print(f"  Arch list:  {arch_list}")
print(f"  sm_120:     {' ' if 'sm_120' in str(arch_list) else '  WARNING'}")

try:
    t = torch.randn(3, 3, device="cuda")
    print(f"  GPU tensor:   (sum={t.sum().item():.2f})")
    print(f"  GPU:        {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  GPU tensor:   {e}")

from torch.utils.cpp_extension import _find_cuda_home
print(f"  CUDA home:  {_find_cuda_home()}")

print("")
print("=== Module Imports ===")

imports = {
    "encode":   "from encode import *",
    "adapters": "from adapters import *",
    "navsim":   "import navsim",
    "navsim.common.dataloader": "from navsim.common.dataloader import SceneLoader",
}

for name, stmt in imports.items():
    try:
        exec(stmt)
        print(f"    {name}")
    except Exception as e:
        short_err = str(e).split('\n')[0][:60]
        print(f"     {name}: {short_err}")

print("")
print("Note: Some import warnings are normal if dependencies have sub-imports.")
PYEOF

echo ""
echo "=========================================="
echo "  SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Environment configured for:"
echo "  GPU:        RTX 5050 Laptop (sm_120, Blackwell)"
echo "  Driver:     580.x"
echo "  CUDA:       13.0"
echo "  PyTorch:    2.10.0+cu130"
echo ""
echo "To activate in new terminals:"
echo "  conda activate l4_env"
echo ""
echo "To build DFA3D:"
echo "  cd $DDPM_ROOT/occlusion/models/encoders/DFA3D"
echo "  bash setup.sh 0"
echo ""
echo "Key paths:"
echo "  DDPM_ROOT:   $DDPM_ROOT"
echo "  CUDA_HOME:   $CUDA_HOME"
echo "  NAVSIM data: $OPENSCENE_DATA_ROOT"
echo ""