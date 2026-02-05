#!/bin/bash
# Fix NAVSIM Python Path for l4_env
# This ensures Python can find the NAVSIM module and navsim_utilize

echo "=========================================="
echo "NAVSIM Python Path Setup"
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

# NAVSIM paths
NAVSIM_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim"
NAVSIM_UTILIZE="$NAVSIM_ROOT/navsim_utilize"
echo "NAVSIM root: $NAVSIM_ROOT"
echo "NAVSIM utilize: $NAVSIM_UTILIZE"

# Check if directories exist
if [ ! -d "$NAVSIM_ROOT/navsim" ]; then
    echo "❌ NAVSIM source not found at: $NAVSIM_ROOT/navsim"
    exit 1
fi

if [ ! -d "$NAVSIM_UTILIZE" ]; then
    echo "❌ navsim_utilize not found at: $NAVSIM_UTILIZE"
    exit 1
fi

echo "✓ NAVSIM source found"
echo "✓ navsim_utilize found"
echo ""

# Method 1: Add to conda environment activation script
echo "=========================================="
echo "Method 1: Auto-activate with conda env"
echo "=========================================="
echo ""

CONDA_ENV_DIR="$CONDA_PREFIX/etc/conda/activate.d"
CONDA_DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
ACTIVATE_SCRIPT="$CONDA_ENV_DIR/navsim_setup.sh"
DEACTIVATE_SCRIPT="$CONDA_DEACTIVATE_DIR/navsim_setup.sh"

# Create directories
mkdir -p "$CONDA_ENV_DIR"
mkdir -p "$CONDA_DEACTIVATE_DIR"

# Create activation script
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# NAVSIM environment setup for l4_env

# Add NAVSIM and navsim_utilize to Python path
export NAVSIM_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim"
export NAVSIM_UTILIZE="$NAVSIM_ROOT/navsim_utilize"
export PYTHONPATH="$NAVSIM_ROOT:$NAVSIM_UTILIZE:$PYTHONPATH"

# NAVSIM data paths
export OPENSCENE_DATA_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download"
export NUPLAN_MAPS_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
export NAVSIM_DEVKIT_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim"
EOF

chmod +x "$ACTIVATE_SCRIPT"

# Create deactivation script
cat > "$DEACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Clean up NAVSIM environment

unset NAVSIM_ROOT
unset NAVSIM_UTILIZE
unset OPENSCENE_DATA_ROOT
unset NUPLAN_MAPS_ROOT
unset NAVSIM_DEVKIT_ROOT

# Remove NAVSIM paths from PYTHONPATH
if [[ -n "$PYTHONPATH" ]]; then
    NAVSIM_PATH="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim"
    NAVSIM_UTIL_PATH="$NAVSIM_PATH/navsim_utilize"
    export PYTHONPATH=$(echo $PYTHONPATH | sed "s|$NAVSIM_UTIL_PATH:||g" | sed "s|:$NAVSIM_UTIL_PATH||g" | sed "s|^$NAVSIM_UTIL_PATH$||g")
    export PYTHONPATH=$(echo $PYTHONPATH | sed "s|$NAVSIM_PATH:||g" | sed "s|:$NAVSIM_PATH||g" | sed "s|^$NAVSIM_PATH$||g")
fi
EOF

chmod +x "$DEACTIVATE_SCRIPT"

echo "✓ Created conda activation scripts:"
echo "  Activate:   $ACTIVATE_SCRIPT"
echo "  Deactivate: $DEACTIVATE_SCRIPT"
echo ""

# Method 2: Add .pth file to site-packages
echo "=========================================="
echo "Method 2: Add .pth file to site-packages"
echo "=========================================="
echo ""

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
PTH_FILE="$SITE_PACKAGES/navsim.pth"

# Add both paths to the .pth file
cat > "$PTH_FILE" << EOF
$NAVSIM_ROOT
$NAVSIM_UTILIZE
EOF

echo "✓ Created .pth file: $PTH_FILE"
echo "  Content:"
cat "$PTH_FILE" | sed 's/^/    /'
echo ""

# Method 3: Install in development mode
echo "=========================================="
echo "Method 3: Install in development mode"
echo "=========================================="
echo ""

cd "$NAVSIM_ROOT"

if [ -f "setup.py" ]; then
    echo "Installing NAVSIM with pip..."
    pip install -e . --no-deps 2>&1 | grep -v "Requirement already satisfied" || true
    
    if [ $? -eq 0 ]; then
        echo "✓ NAVSIM installed in development mode"
    else
        echo "⚠️  Installation had issues, but path methods should still work"
    fi
else
    echo "⚠️  No setup.py found, skipping pip install"
fi

echo ""
echo "=========================================="
echo "Verifying Setup"
echo "=========================================="
echo ""

# Apply the environment now
source "$ACTIVATE_SCRIPT"

echo "Current PYTHONPATH:"
echo "$PYTHONPATH" | tr ':' '\n' | head -7
echo ""

echo "Testing imports..."

# Test basic import
python << 'PYEOF'
import sys
print("Python path (first 7):")
for i, path in enumerate(sys.path[:7], 1):
    print(f"  {i}. {path}")
print()

# Test NAVSIM import
try:
    import navsim
    print("✓ import navsim - SUCCESS")
    print(f"  Location: {navsim.__file__}")
except ImportError as e:
    print(f"❌ import navsim - FAILED: {e}")
    sys.exit(1)

# Test navsim_utilize imports
try:
    from contract.data_contract import FeatureType, DataContract
    print("✓ import contract.data_contract - SUCCESS")
except ImportError as e:
    print(f"❌ import contract.data_contract - FAILED: {e}")
    print("  Make sure navsim_utilize is in PYTHONPATH")
    sys.exit(1)

try:
    from datasets.base import BaseNavsimDataset
    print("✓ import datasets.base - SUCCESS")
except ImportError as e:
    print(f"❌ import datasets.base - FAILED: {e}")
    sys.exit(1)

# Test specific NAVSIM imports
imports_to_test = [
    "navsim.common.dataloader",
    "navsim.common.dataclasses",
]

all_success = True
for module_name in imports_to_test:
    try:
        __import__(module_name)
        print(f"✓ import {module_name} - SUCCESS")
    except ImportError as e:
        print(f"❌ import {module_name} - FAILED: {e}")
        all_success = False

if all_success:
    print("\n✓ All imports successful!")
else:
    print("\n⚠️  Some imports failed")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "NAVSIM and navsim_utilize are now accessible in your l4_env environment."
    echo ""
    echo "To apply changes:"
    echo "  conda deactivate"
    echo "  conda activate l4_env"
    echo ""
    echo "Or to use immediately in current shell:"
    echo "  source $ACTIVATE_SCRIPT"
    echo ""
    echo "Now you can run scripts with absolute imports:"
    echo "  cd ~/DPJI/transdiffuser/DDPM/datasets/navsim/navsim_utilize/datasets"
    echo "  python test_base_dataset.py"
    echo ""
    echo "Or from any directory:"
    echo "  python -m datasets.test_base_dataset"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Setup verification failed"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo ""
    echo "1. Check if directory structure is correct:"
    echo "   ls -la $NAVSIM_ROOT/navsim/"
    echo "   ls -la $NAVSIM_UTILIZE/"
    echo "   # Should see __init__.py files"
    echo ""
    echo "2. Manually set PYTHONPATH and test:"
    echo "   export PYTHONPATH=$NAVSIM_ROOT:$NAVSIM_UTILIZE:\$PYTHONPATH"
    echo "   python -c 'import navsim; print(navsim.__file__)'"
    echo "   python -c 'from contract.data_contract import FeatureType'"
    echo ""
    echo "3. Check Python path:"
    echo "   python -c 'import sys; print(sys.path)'"
    echo ""
    exit 1
fi