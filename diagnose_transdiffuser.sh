#!/bin/bash
# Diagnostic script to check transdiffuser setup

echo "=========================================="
echo "Transdiffuser Environment Diagnostics"
echo "=========================================="
echo ""

# Check conda environment
echo "1. Conda Environment:"
echo "   Current: $CONDA_DEFAULT_ENV"
echo "   Prefix: $CONDA_PREFIX"
echo ""

# Check Python
echo "2. Python:"
which python
python --version
echo ""

# Check if activation script exists
echo "3. Activation Scripts:"
ACTIVATE_SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/transdiffuser_setup.sh"
DEACTIVATE_SCRIPT="$CONDA_PREFIX/etc/conda/deactivate.d/transdiffuser_setup.sh"

if [ -f "$ACTIVATE_SCRIPT" ]; then
    echo "   ✓ Activation script exists"
    echo "     Location: $ACTIVATE_SCRIPT"
else
    echo "   ❌ Activation script NOT found"
    echo "     Expected: $ACTIVATE_SCRIPT"
fi

if [ -f "$DEACTIVATE_SCRIPT" ]; then
    echo "   ✓ Deactivation script exists"
else
    echo "   ❌ Deactivation script NOT found"
fi
echo ""

# Check .pth file
echo "4. PTH File:"
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
PTH_FILE="$SITE_PACKAGES/transdiffuser.pth"

if [ -f "$PTH_FILE" ]; then
    echo "   ✓ PTH file exists"
    echo "     Location: $PTH_FILE"
    echo "     Content:"
    cat "$PTH_FILE" | sed 's/^/       /'
else
    echo "   ❌ PTH file NOT found"
    echo "     Expected: $PTH_FILE"
fi
echo ""

# Check environment variables
echo "5. Environment Variables:"
echo "   PROJECT_ROOT: ${PROJECT_ROOT:-NOT SET}"
echo "   DDPM_ROOT: ${DDPM_ROOT:-NOT SET}"
echo "   NAVSIM_ROOT: ${NAVSIM_ROOT:-NOT SET}"
echo ""

# Check PYTHONPATH
echo "6. PYTHONPATH (first 10 entries):"
if [[ -n "$PYTHONPATH" ]]; then
    echo "$PYTHONPATH" | tr ':' '\n' | head -10 | nl
else
    echo "   PYTHONPATH is empty or not set"
fi
echo ""

# Check Python sys.path
echo "7. Python sys.path (first 10 entries):"
python << 'PYEOF'
import sys
for i, path in enumerate(sys.path[:10], 1):
    print(f"   {i}. {path}")
PYEOF
echo ""

# Check project structure
echo "8. Project Structure:"
PROJECT_ROOT="/home/phamtamadas/DPJI/transdiffuser"
DDPM_ROOT="$PROJECT_ROOT/DDPM"

dirs_to_check=(
    "$PROJECT_ROOT"
    "$DDPM_ROOT"
    "$DDPM_ROOT/encode"
    "$DDPM_ROOT/adapters"
    "$DDPM_ROOT/datasets"
    "$DDPM_ROOT/datasets/navsim"
    "$DDPM_ROOT/datasets/navsim/navsim_utilize"
)

for dir in "${dirs_to_check[@]}"; do
    if [ -d "$dir" ]; then
        has_init=""
        if [ -f "$dir/__init__.py" ]; then
            has_init="(has __init__.py)"
        else
            has_init="(MISSING __init__.py)"
        fi
        echo "   ✓ $dir $has_init"
    else
        echo "   ❌ $dir (NOT FOUND)"
    fi
done
echo ""

# Test imports
echo "9. Import Tests:"
python << 'PYEOF'
import sys

test_imports = [
    ("encode", "import encode"),
    ("adapters", "import adapters"),
    ("datasets", "import datasets"),
    ("navsim", "import navsim"),
    ("contract", "from contract.data_contract import FeatureType"),
]

for name, import_cmd in test_imports:
    try:
        exec(import_cmd)
        print(f"   ✓ {import_cmd}")
    except ImportError as e:
        print(f"   ❌ {import_cmd}")
        print(f"      Error: {str(e)[:80]}")
    except Exception as e:
        print(f"   ⚠️  {import_cmd}")
        print(f"      Error: {str(e)[:80]}")
PYEOF
echo ""

echo "=========================================="
echo "Diagnostics Complete"
echo "=========================================="
echo ""
echo "If you see issues above, try:"
echo "  1. Deactivate and reactivate: conda deactivate && conda activate l4_env"
echo "  2. Run the setup script again: bash setup_transdiffuser_fixed.sh"
echo "  3. Check that all directories exist and have __init__.py files"
echo ""