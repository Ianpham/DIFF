#!/bin/bash
# Complete Transdiffuser Project Setup for l4_env
# Run this ONCE to make the entire project importable from anywhere

echo "=========================================="
echo "Transdiffuser Project Setup"
echo "=========================================="
echo ""

# Check current environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ No conda environment active"
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
    echo "❌ Project root not found at: $PROJECT_ROOT"
    exit 1
fi

if [ ! -d "$DDPM_ROOT" ]; then
    echo "❌ DDPM directory not found at: $DDPM_ROOT"
    exit 1
fi

echo "✓ Project structure found"
echo ""

# Ensure all directories have __init__.py files
echo "=========================================="
echo "Creating __init__.py files"
echo "=========================================="
echo ""

# Create __init__.py in key directories
touch "$DDPM_ROOT/__init__.py"
touch "$DDPM_ROOT/encode/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/adapters/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/datasets/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/datasets/navsim/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/datasets/navsim/navsim_utilize/__init__.py" 2>/dev/null || true

echo "✓ Created __init__.py files"
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

# Create directories
mkdir -p "$CONDA_ENV_DIR"
mkdir -p "$CONDA_DEACTIVATE_DIR"

# Create comprehensive activation script
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Transdiffuser Complete Environment Setup

# Main project paths
export PROJECT_ROOT="/home/phamtamadas/DPJI/transdiffuser"
export DDPM_ROOT="$PROJECT_ROOT/DDPM"

# Key module paths
export ENCODE_ROOT="$DDPM_ROOT/encode"
export ADAPTERS_ROOT="$DDPM_ROOT/adapters"
export DATASETS_ROOT="$DDPM_ROOT/datasets"

# NAVSIM specific paths
export NAVSIM_ROOT="$DATASETS_ROOT/navsim"
export NAVSIM_UTILIZE="$NAVSIM_ROOT/navsim_utilize"
export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"

# NAVSIM data paths
export OPENSCENE_DATA_ROOT="$NAVSIM_ROOT/download"
export NUPLAN_MAPS_ROOT="$NAVSIM_ROOT/download/maps"

# Build comprehensive PYTHONPATH
# Priority order (first paths searched first):
# 1. DDPM_ROOT - for top-level imports: from encode.X, from adapters.X, from datasets.X
# 2. NAVSIM_ROOT - for: import navsim
# 3. NAVSIM_UTILIZE - for: from contract.X, from datasets.base
export PYTHONPATH="$DDPM_ROOT:$NAVSIM_ROOT:$NAVSIM_UTILIZE:$PYTHONPATH"

# Optional: Set working directory hint
export TRANSDIFFUSER_HOME="$DDPM_ROOT"
EOF

chmod +x "$ACTIVATE_SCRIPT"

# Create deactivation script
cat > "$DEACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Clean up Transdiffuser environment

# Unset all custom environment variables
unset PROJECT_ROOT
unset DDPM_ROOT
unset ENCODE_ROOT
unset ADAPTERS_ROOT
unset DATASETS_ROOT
unset NAVSIM_ROOT
unset NAVSIM_UTILIZE
unset NAVSIM_DEVKIT_ROOT
unset OPENSCENE_DATA_ROOT
unset NUPLAN_MAPS_ROOT
unset TRANSDIFFUSER_HOME

# Remove custom paths from PYTHONPATH
if [[ -n "$PYTHONPATH" ]]; then
    DDPM_PATH="/home/phamtamadas/DPJI/transdiffuser/DDPM"
    NAVSIM_PATH="$DDPM_PATH/datasets/navsim"
    NAVSIM_UTIL_PATH="$NAVSIM_PATH/navsim_utilize"
    
    export PYTHONPATH=$(echo $PYTHONPATH | sed "s|$DDPM_PATH:||g" | sed "s|:$DDPM_PATH||g" | sed "s|^$DDPM_PATH$||g")
    export PYTHONPATH=$(echo $PYTHONPATH | sed "s|$NAVSIM_PATH:||g" | sed "s|:$NAVSIM_PATH||g" | sed "s|^$NAVSIM_PATH$||g")
    export PYTHONPATH=$(echo $PYTHONPATH | sed "s|$NAVSIM_UTIL_PATH:||g" | sed "s|:$NAVSIM_UTIL_PATH||g" | sed "s|^$NAVSIM_UTIL_PATH$||g")
fi
EOF

chmod +x "$DEACTIVATE_SCRIPT"

echo "✓ Created conda environment hooks"
echo ""

# Add .pth file for site-packages
echo "=========================================="
echo "Setting up .pth file"
echo "=========================================="
echo ""

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
PTH_FILE="$SITE_PACKAGES/transdiffuser.pth"

cat > "$PTH_FILE" << EOF
$DDPM_ROOT
$DDPM_ROOT/datasets/navsim
$DDPM_ROOT/datasets/navsim/navsim_utilize
EOF

echo "✓ Created .pth file: $PTH_FILE"
echo "  Content:"
cat "$PTH_FILE" | sed 's/^/    /'
echo ""

# Apply environment immediately
echo "=========================================="
echo "Applying environment"
echo "=========================================="
echo ""

source "$ACTIVATE_SCRIPT"

echo "Current PYTHONPATH (first 5 entries):"
echo "$PYTHONPATH" | tr ':' '\n' | head -5
echo ""

# Comprehensive verification
echo "=========================================="
echo "Verifying Setup"
echo "=========================================="
echo ""

python << 'PYEOF'
import sys
import os

print("=" * 50)
print("Python Environment Check")
print("=" * 50)
print()

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print()

print("PYTHONPATH entries (first 8):")
for i, path in enumerate(sys.path[:8], 1):
    print(f"  {i}. {path}")
print()

# Test all key imports
imports_to_test = [
    ("encode", "from encode.requirements import EncoderRequirements"),
    ("adapters", "import adapters"),
    ("datasets", "import datasets"),
    ("navsim", "import navsim"),
    ("contract", "from contract.data_contract import FeatureType"),
    ("navsim_utilize.datasets", "from datasets.base import BaseNavsimDataset"),
]

print("=" * 50)
print("Testing Imports")
print("=" * 50)
print()

all_success = True
for module_name, import_statement in imports_to_test:
    try:
        exec(import_statement)
        print(f"✓ {import_statement}")
    except ImportError as e:
        print(f"❌ {import_statement}")
        print(f"   Error: {e}")
        all_success = False
    except Exception as e:
        print(f"⚠️  {import_statement}")
        print(f"   Error: {e}")

print()

if all_success:
    print("=" * 50)
    print("✓ ALL IMPORTS SUCCESSFUL!")
    print("=" * 50)
    sys.exit(0)
else:
    print("=" * 50)
    print("⚠️  SOME IMPORTS FAILED")
    print("=" * 50)
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ SETUP COMPLETE!"
    echo "=========================================="
    echo ""
    echo "The entire transdiffuser project is now importable!"
    echo ""
    echo "To activate in new terminals:"
    echo "  conda activate l4_env"
    echo ""
    echo "To use immediately:"
    echo "  source $ACTIVATE_SCRIPT"
    echo ""
    echo "Now you can work from ANY directory:"
    echo ""
    echo "  # Work in adapters"
    echo "  cd ~/DPJI/transdiffuser/DDPM/adapters"
    echo "  python test_adapters.py"
    echo ""
    echo "  # Work in encode"
    echo "  cd ~/DPJI/transdiffuser/DDPM/encode"
    echo "  python test_encoder.py"
    echo ""
    echo "  # Work from project root"
    echo "  cd ~/DPJI/transdiffuser/DDPM"
    echo "  python -m adapters.test_adapters"
    echo ""
    echo "All imports will work consistently:"
    echo "  from encode.requirements import EncoderRequirements"
    echo "  from adapters.some_module import SomeClass"
    echo "  from datasets.navsim_dataset import NavsimDataset"
    echo "  import navsim"
    echo "  from contract.data_contract import FeatureType"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Setup verification failed"
    echo "=========================================="
    echo ""
    echo "Please check:"
    echo "1. Project structure at: $PROJECT_ROOT"
    echo "2. Python environment: conda activate l4_env"
    echo "3. Manual test: python -c 'import sys; print(sys.path)'"
    echo ""
    exit 1
fi