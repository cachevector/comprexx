#!/usr/bin/env bash
#
# Build and publish comprexx to PyPI.
#
# Usage:
#   scripts/publish.sh              # build + check, then prompt before upload
#   scripts/publish.sh --test       # upload to TestPyPI instead of PyPI
#   scripts/publish.sh --skip-tests # skip the test suite
#   scripts/publish.sh --yes        # skip the upload confirmation prompt
#
# Requires ~/.pypirc with [pypi] (and optionally [testpypi]) sections.

set -euo pipefail

REPO="pypi"
RUN_TESTS=1
ASSUME_YES=0

for arg in "$@"; do
    case "$arg" in
        --test) REPO="testpypi" ;;
        --skip-tests) RUN_TESTS=0 ;;
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

# Move to repo root (parent of this script)
cd "$(dirname "$0")/.."

if [ ! -f pyproject.toml ]; then
    echo "error: pyproject.toml not found, are you in the repo root?" >&2
    exit 1
fi

VERSION=$(grep -E '^version' pyproject.toml | head -1 | sed -E 's/version *= *"([^"]+)"/\1/')
echo "==> comprexx version: $VERSION"
echo "==> target index:     $REPO"

# Activate venv if present
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo
echo "==> Ensuring build and twine are installed"
if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --upgrade build twine
elif python -m pip --version >/dev/null 2>&1; then
    python -m pip install --quiet --upgrade build twine
else
    python -m ensurepip --upgrade
    python -m pip install --quiet --upgrade build twine
fi

if [ "$RUN_TESTS" -eq 1 ]; then
    echo
    echo "==> Running test suite"
    pytest -q
fi

echo
echo "==> Cleaning old build artifacts"
rm -rf dist/ build/ ./*.egg-info

echo
echo "==> Building sdist and wheel"
python -m build

echo
echo "==> Validating package metadata"
twine check dist/*

echo
echo "==> Build artifacts:"
ls -lh dist/

if [ "$ASSUME_YES" -ne 1 ]; then
    echo
    read -r -p "Upload comprexx $VERSION to $REPO? [y/N] " confirm
    case "$confirm" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

echo
echo "==> Uploading to $REPO"
twine upload --repository "$REPO" dist/*

echo
echo "==> Done. Verify at:"
if [ "$REPO" = "testpypi" ]; then
    echo "    https://test.pypi.org/project/comprexx/$VERSION/"
else
    echo "    https://pypi.org/project/comprexx/$VERSION/"
fi
