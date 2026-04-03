#!/bin/bash
# Clean up corrupted PYTHONPATH entries

echo "=========================================="
echo "Cleaning PYTHONPATH Corruption"
echo "=========================================="
echo ""

if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ No conda environment active"
    echo "Please run: conda activate l4_env"
    exit 1
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Show current PYTHONPATH
echo "Before cleanup - PYTHONPATH entries:"
echo "$PYTHONPATH" | tr ':' '\n' | nl | head -10
echo ""

# Clean PYTHONPATH by removing corrupted/recursive entries
# Keep only the valid paths
DDPM_ROOT="/home/phamtamadas/DPJI/transdiffuser/DDPM"
NAVSIM_ROOT="$DDPM_ROOT/datasets/navsim"
NAVSIM_UTILIZE="$NAVSIM_ROOT/navsim_utilize"

# Build clean PYTHONPATH with only valid entries
NEW_PYTHONPATH="$DDPM_ROOT:$NAVSIM_ROOT:$NAVSIM_UTILIZE"

# Preserve any other valid paths that aren't our custom ones
IFS=':' read -ra PATH_ARRAY <<< "$PYTHONPATH"
for path in "${PATH_ARRAY[@]}"; do
    # Skip our custom paths (already added)
    if [[ "$path" == "$DDPM_ROOT" ]] || \
       [[ "$path" == "$NAVSIM_ROOT" ]] || \
       [[ "$path" == "$NAVSIM_UTILIZE" ]]; then
        continue
    fi
    
    # Skip corrupted paths (those with repeated /datasets/navsim)
    if [[ "$path" == *"/datasets/navsim/datasets/navsim"* ]]; then
        echo "Removing corrupted path: ${path:0:80}..."
        continue
    fi
    
    # Keep other valid paths
    if [[ -d "$path" ]] && [[ -n "$path" ]]; then
        NEW_PYTHONPATH="$NEW_PYTHONPATH:$path"
    fi
done

export PYTHONPATH="$NEW_PYTHONPATH"

echo "After cleanup - PYTHONPATH entries:"
echo "$PYTHONPATH" | tr ':' '\n' | nl
echo ""

# Update the activation script to prevent future corruption
ACTIVATE_SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/transdiffuser_setup.sh"

if [ -f "$ACTIVATE_SCRIPT" ]; then
    echo "Updating activation script to prevent future corruption..."
    
    cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Transdiffuser Complete Environment Setup - Clean Version

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

# Optional: Set working directory hint
export TRANSDIFFUSER_HOME="$DDPM_ROOT"

# Build clean PYTHONPATH - only add if not already present
# This prevents duplicate/recursive paths

# Remove any existing transdiffuser paths first
if [[ -n "$PYTHONPATH" ]]; then
    # Remove our paths if they exist
    PYTHONPATH=$(echo "$PYTHONPATH" | sed "s|$DDPM_ROOT:||g" | sed "s|:$DDPM_ROOT||g" | sed "s|^$DDPM_ROOT$||")
    PYTHONPATH=$(echo "$PYTHONPATH" | sed "s|$NAVSIM_ROOT:||g" | sed "s|:$NAVSIM_ROOT||g" | sed "s|^$NAVSIM_ROOT$||")
    PYTHONPATH=$(echo "$PYTHONPATH" | sed "s|$NAVSIM_UTILIZE:||g" | sed "s|:$NAVSIM_UTILIZE||g" | sed "s|^$NAVSIM_UTILIZE$||")
    
    # Remove any corrupted recursive paths
    PYTHONPATH=$(echo "$PYTHONPATH" | sed 's|/datasets/navsim/datasets/navsim[^:]*||g')
fi

# Now add clean paths at the beginning
export PYTHONPATH="$DDPM_ROOT:$NAVSIM_ROOT:$NAVSIM_UTILIZE:$PYTHONPATH"

# Clean up any trailing/leading colons
export PYTHONPATH=$(echo "$PYTHONPATH" | sed 's/^://; s/:$//')
EOF
    
    chmod +x "$ACTIVATE_SCRIPT"
    echo "✓ Activation script updated"
else
    echo "  Activation script not found, skipping update"
fi

echo ""
echo "=========================================="
echo "✓ Cleanup Complete!"
echo "=========================================="
echo ""
echo "To make this permanent:"
echo "  1. Deactivate: conda deactivate"
echo "  2. Reactivate: conda activate l4_env"
echo ""
echo "The updated activation script will prevent future corruption."
echo ""

# Verify imports still work
echo "Verifying imports still work..."
python << 'PYEOF'
test_imports = [
    ("encode", "import encode"),
    ("adapters", "import adapters"),
    ("datasets", "import datasets"),
    ("navsim", "import navsim"),
    ("contract", "from contract.data_contract import FeatureType"),
]

all_ok = True
for name, import_cmd in test_imports:
    try:
        exec(import_cmd)
        print(f"  ✓ {import_cmd}")
    except Exception as e:
        print(f"   {import_cmd}: {e}")
        all_ok = False

if all_ok:
    print("\n✓ All imports working!")
else:
    print("\n  Some imports failed")
PYEOF