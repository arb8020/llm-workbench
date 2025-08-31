#!/bin/bash
# deploy_runpod.sh - Package and deploy to RunPod

set -e

echo "Creating git bundle with all changes..."
git bundle create inference.bundle HEAD --all

echo "Bundle created: inference.bundle"
echo ""
echo "To deploy to RunPod:"
echo "1. Upload inference.bundle to your RunPod instance"
echo "2. Run the following commands in RunPod terminal:"
echo ""
echo "   git clone inference.bundle inference"
echo "   cd inference"
echo "   pip install -e .[jax]  # or [torch]"
echo "   python jax/utils/test_core.py"
echo ""
echo "To update on RunPod after local changes:"
echo "1. Run this script again to create new bundle"
echo "2. Upload new bundle to RunPod"
echo "3. In RunPod: git pull inference.bundle"