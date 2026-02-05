#!/bin/bash
# Fix Encode Module Python Path for l4_env
# This ensures Python can find the encode module alongside NAVSIM

echo "=========================================="
echo "Encode Module Python Path Setup"
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

# Encode module paths
TRANSDIFFUSER_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM"
ENCODE_ROOT="$TRANSDIFFUSER_ROOT/encode"
DATASETS_ROOT="$TRANSDIFFUSER_ROOT/datasets"

echo "Transdiffuser root: $TRANSDIFFUSER_ROOT"
echo "Encode root: $ENCODE_ROOT"
echo "Datasets root: $DATASETS_ROOT"
echo ""

# Check if directories exist
if [ ! -d "$ENCODE_ROOT" ]; then
    echo "❌ Encode module not found at: $ENCODE_ROOT"
    exit 1
fi

if [ ! -d "$DATASETS_ROOT" ]; then
    echo "❌ Datasets not found at: $DATASETS_ROOT"
    exit 1
fi

echo "✓ Encode module found"
echo "✓ Datasets found"
echo ""

# Ensure __init__.py exists in encode directory
if [ ! -f "$ENCODE_ROOT/__init__.py" ]; then
    echo "Creating __init__.py in encode directory..."
    touch "$ENCODE_ROOT/__init__.py"
    echo "✓ Created $ENCODE_ROOT/__init__.py"
fi

# Method 1: Update conda environment activation script
echo "=========================================="
echo "Method 1: Update conda env activation"
echo "=========================================="
echo ""

CONDA_ENV_DIR="$CONDA_PREFIX/etc/conda/activate.d"
CONDA_DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
ACTIVATE_SCRIPT="$CONDA_ENV_DIR/navsim_setup.sh"
DEACTIVATE_SCRIPT="$CONDA_DEACTIVATE_DIR/navsim_setup.sh"

# Check if NAVSIM activation script exists
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo "❌ NAVSIM activation script not found at: $ACTIVATE_SCRIPT"
    echo "Please run the NAVSIM setup script first."
    exit 1
fi

echo "Updating existing NAVSIM activation script..."

# Backup original
cp "$ACTIVATE_SCRIPT" "${ACTIVATE_SCRIPT}.backup"

# Update activation script to include encode and datasets paths
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# NAVSIM and Encode environment setup for l4_env

# NAVSIM paths
export NAVSIM_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim"
export NAVSIM_UTILIZE="$NAVSIM_ROOT/navsim_utilize"

# Transdiffuser/Encode paths
export TRANSDIFFUSER_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM"
export ENCODE_ROOT="$TRANSDIFFUSER_ROOT/encode"
export DATASETS_ROOT="$TRANSDIFFUSER_ROOT/datasets"
export NAVSIM_DATASETS_ROOT="$DATASETS_ROOT/navsim"

# Add all paths to PYTHONPATH
# Add TRANSDIFFUSER_ROOT for: from encode.X, from datasets.X
# Add NAVSIM_ROOT for: import navsim
# Add NAVSIM_UTILIZE for: from contract.X, from datasets.base
# Add NAVSIM_DATASETS_ROOT for: from navsim_utilize.X
export PYTHONPATH="$TRANSDIFFUSER_ROOT:$NAVSIM_ROOT:$NAVSIM_UTILIZE:$NAVSIM_DATASETS_ROOT:$PYTHONPATH"

# NAVSIM data paths
export OPENSCENE_DATA_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download"
export NUPLAN_MAPS_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
export NAVSIM_DEVKIT_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim"
EOF

chmod +x "$ACTIVATE_SCRIPT"

# Update deactivation script
cat > "$DEACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Clean up NAVSIM and Encode environment

unset NAVSIM_ROOT
unset NAVSIM_UTILIZE
unset TRANSDIFFUSER_ROOT
unset ENCODE_ROOT
unset DATASETS_ROOT
unset NAVSIM_DATASETS_ROOT
unset OPENSCENE_DATA_ROOT
unset NUPLAN_MAPS_ROOT
unset NAVSIM_DEVKIT_ROOT

# Remove all custom paths from PYTHONPATH
if [[ -n "$PYTHONPATH" ]]; then
    TRANSDIFF_PATH="/home/phamtamadas/DPJI/transdiffuser/DDPM"
    NAVSIM_PATH="$TRANSDIFF_PATH/datasets/navsim"
    NAVSIM_UTIL_PATH="$NAVSIM_PATH/navsim_utilize"
    NAVSIM_DATASETS_PATH="$TRANSDIFF_PATH/datasets/navsim"
    
    # Remove each path
    export PYTHONPATH=$(echo $PYTHONPATH | sed "s|$TRANSDIFF_PATH:||g" | sed "s|:$TRANSDIFF_PATH||g" | sed "s|^$TRANSDIFF_PATH$||g")
    export PYTHONPATH=$(echo $PYTHONPATH | sed "s|$NAVSIM_DATASETS_PATH:||g" | sed "s|:$NAVSIM_DATASETS_PATH||g" | sed "s|^$NAVSIM_DATASETS_PATH$||g")
    export PYTHONPATH=$(echo $PYTHONPATH | sed "s|$NAVSIM_UTIL_PATH:||g" | sed "s|:$NAVSIM_UTIL_PATH||g" | sed "s|^$NAVSIM_UTIL_PATH$||g")
    export PYTHONPATH=$(echo $PYTHONPATH | sed "s|$NAVSIM_PATH:||g" | sed "s|:$NAVSIM_PATH||g" | sed "s|^$NAVSIM_PATH$||g")
fi
EOF

chmod +x "$DEACTIVATE_SCRIPT"

echo "✓ Updated conda activation scripts"
echo "  Activate:   $ACTIVATE_SCRIPT"
echo "  Deactivate: $DEACTIVATE_SCRIPT"
echo ""

# Method 2: Update .pth file
echo "=========================================="
echo "Method 2: Update .pth file"
echo "=========================================="
echo ""

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
PTH_FILE="$SITE_PACKAGES/navsim.pth"

# Update .pth file to include transdiffuser root and navsim datasets root
cat > "$PTH_FILE" << EOF
$TRANSDIFFUSER_ROOT
$DATASETS_ROOT/navsim
$DATASETS_ROOT/navsim/navsim_utilize
EOF

echo "✓ Updated .pth file: $PTH_FILE"
echo "  Content:"
cat "$PTH_FILE" | sed 's/^/    /'
echo ""

# Apply the environment now
echo "=========================================="
echo "Applying Environment Changes"
echo "=========================================="
echo ""

source "$ACTIVATE_SCRIPT"

echo "Current PYTHONPATH:"
echo "$PYTHONPATH" | tr ':' '\n' | head -10
echo ""

# Verify setup
echo "=========================================="
echo "Verifying Setup"
echo "=========================================="
echo ""

python << 'PYEOF'
import sys
print("Python path (first 10 entries):")
for i, path in enumerate(sys.path[:10], 1):
    print(f"  {i}. {path}")
print()

# Test encode import
try:
    import encode
    print("✓ import encode - SUCCESS")
    if hasattr(encode, '__file__'):
        print(f"  Location: {encode.__file__}")
    else:
        print(f"  Location: {encode.__path__[0]}")
except ImportError as e:
    print(f"❌ import encode - FAILED: {e}")
    print("\nDEBUG: Current sys.path:")
    for p in sys.path[:5]:
        print(f"  {p}")
    sys.exit(1)

# Test encode.requirements import
try:
    from encode.requirements import EncoderRequirements
    print("✓ import encode.requirements.EncoderRequirements - SUCCESS")
except ImportError as e:
    print(f"❌ import encode.requirements - FAILED: {e}")
    sys.exit(1)

# Test datasets import
try:
    from datasets import NavsimDataset
    print("✓ import datasets.NavsimDataset - SUCCESS")
except ImportError as e:
    print(f"⚠️  import datasets.NavsimDataset - FAILED: {e}")
    print("  (This may be expected if NavsimDataset isn't set up yet)")

# Test full path import
try:
    from datasets.navsim.navsim_utilize.datasets.navsim_basic import NavsimDataset as NavsimBasic
    print("✓ import datasets.navsim.navsim_utilize.datasets.navsim_basic.NavsimDataset - SUCCESS")
except ImportError as e:
    print(f"⚠️  Full path NavsimDataset import - FAILED: {e}")

# Test NAVSIM imports (should still work)
try:
    import navsim
    print("✓ import navsim - SUCCESS")
except ImportError as e:
    print(f"⚠️  import navsim - FAILED: {e}")

try:
    from contract.data_contract import FeatureType
    print("✓ import contract.data_contract.FeatureType - SUCCESS")
except ImportError as e:
    print(f"⚠️  import FeatureType from contract - FAILED: {e}")

print("\n✓ Encode module is now importable!")
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "The encode module is now accessible in your l4_env environment."
    echo ""
    echo "To apply changes in new terminals:"
    echo "  conda deactivate"
    echo "  conda activate l4_env"
    echo ""
    echo "Or to use immediately in current shell:"
    echo "  source $ACTIVATE_SCRIPT"
    echo ""
    echo "Now you can run your test script:"
    echo "  cd ~/DPJI/transdiffuser/DDPM/encode"
    echo "  python test_encoder_requirements.py"
    echo ""
    echo "Import structure:"
    echo "  from encode.requirements import EncoderRequirements"
    echo "  from contract.data_contract import FeatureType"
    echo "  from datasets.navsim.navsim_utilize.datasets.navsim_basic import NavsimDataset"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Setup verification failed"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo ""
    echo "1. Ensure __init__.py exists in encode directory:"
    echo "   ls -la $ENCODE_ROOT/__init__.py"
    echo ""
    echo "2. Check directory structure:"
    echo "   ls -la $TRANSDIFFUSER_ROOT/"
    echo "   # Should see: encode/, datasets/"
    echo ""
    echo "3. Manually test PYTHONPATH:"
    echo "   export PYTHONPATH=$TRANSDIFFUSER_ROOT:\$PYTHONPATH"
    echo "   python -c 'import encode; print(encode)'"
    echo ""
    echo "4. Check if NAVSIM setup was run first:"
    echo "   python -c 'import navsim; print(navsim.__file__)'"
    echo ""
    exit 1
fi