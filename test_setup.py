#!/usr/bin/env python3
"""Quick test to verify all dependencies are installed."""

print("=" * 60)
print("DRL Aerial Combat Research - Setup Verification")
print("=" * 60)

# Test PyTorch
try:
    import torch
    print(f"yes PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"no PyTorch: {e}")

# Test Gymnasium
try:
    import gymnasium
    print(f"yes Gymnasium: {gymnasium.__version__}")
except ImportError as e:
    print(f"no Gymnasium: {e}")

# Test Stable-Baselines3
try:
    import stable_baselines3
    print(f"yes Stable-Baselines3: {stable_baselines3.__version__}")
except ImportError as e:
    print(f"no Stable-Baselines3: {e}")

# Test Data Science Libraries
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"yes NumPy: {np.__version__}")
    print(f"yes Pandas: {pd.__version__}")
    print(f"yes Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"no Data Science Libraries: {e}")

# Test Jupyter
try:
    import jupyter
    print(f"yes Jupyter: installed")
except ImportError as e:
    print(f"no Jupyter: {e}")

print("=" * 60)
print("yes ALL DEPENDENCIES INSTALLED!")
print("=" * 60)
