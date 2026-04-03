#!/bin/bash
# Complete Transdiffuser Project Setup for l4_env
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

# Create __init__.py in key directories
touch "$DDPM_ROOT/__init__.py"
touch "$DDPM_ROOT/encode/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/adapters/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/datasets/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/datasets/navsim/__init__.py" 2>/dev/null || true
touch "$DDPM_ROOT/datasets/navsim/navsim_utilize/__init__.py" 2>/dev/null || true

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

echo "  Created conda environment hooks"
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

echo "Current PYTHONPATH (first 5 entries):"
echo "$PYTHONPATH" | tr ':' '\n' | head -5
echo ""

# Simplified verification that won't crash
echo "=========================================="
echo "Verifying Setup"
echo "=========================================="
echo ""

# Test Python can run
python -c "import sys; print('Python OK:', sys.version.split()[0])" || {
    echo "  Python test failed"
    exit 1
}

# Test sys.path setup
python -c "import sys; print('PYTHONPATH entries:', len(sys.path))" || {
    echo "  PYTHONPATH test failed"
    exit 1
}

# Try imports but don't fail if they don't work yet
echo "Testing key imports (failures are OK for now)..."
echo ""

python << 'PYEOF'
import sys

imports_to_test = [
    "encode",
    "adapters", 
    "datasets",
    "navsim",
]

print("Attempting imports:")
for module_name in imports_to_test:
    try:
        __import__(module_name)
        print(f"    {module_name}")
    except Exception as e:
        print(f"     {module_name}: {str(e)[:50]}")

print("\nNote: Some import failures are expected if modules have dependencies.")
print("The PYTHONPATH setup is complete - you can now test your specific imports.")
PYEOF

echo ""
echo "=========================================="
echo "  SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "The transdiffuser environment is configured!"
echo ""
echo "To activate in new terminals:"
echo "  conda activate l4_env"
echo ""
echo "The environment will activate automatically with these paths:"
echo "  - DDPM_ROOT: $DDPM_ROOT"
echo "  - NAVSIM_ROOT: $DDPM_ROOT/datasets/navsim"
echo "  - NAVSIM_UTILIZE: $DDPM_ROOT/datasets/navsim/navsim_utilize"
echo ""
echo "Test your imports manually:"
echo "  python -c 'from encode.requirements import EncoderRequirements'"
echo "  python -c 'import navsim'"
echo "  python -c 'from contract.data_contract import FeatureType'"
echo ""
echo "If imports fail, check:"
echo "  1. Do the modules exist at the expected paths?"
echo "  2. Do they have any missing dependencies?"
echo "  3. Run: python -c 'import sys; print(sys.path)' to verify paths"
echo ""